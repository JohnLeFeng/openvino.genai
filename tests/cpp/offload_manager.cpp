// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "continuous_batching/cache/offload_manager.hpp"
#include "openvino/genai/scheduler_config.hpp"

using ov::genai::ICacheManager;
using ov::genai::OffloadConfig;
using ov::genai::OffloadManager;

namespace {

// Minimal in-memory cache manager standing in for a real device KV cache. Each "device block" is a
// fixed-size byte buffer; read/write copy whole blocks to/from a host buffer the same way the real
// KVCacheManager serializes a block across all decoder layers.
class FakeCacheManager : public ICacheManager {
public:
    FakeCacheManager(size_t num_blocks, size_t block_size_in_bytes)
        : m_block_size_in_bytes(block_size_in_bytes),
          m_blocks(num_blocks, std::vector<uint8_t>(block_size_in_bytes, 0)) {}

    void allocate_cache_if_needed(size_t) override {}
    void copy_blocks(const std::map<size_t, std::list<size_t>>&) override {}
    void clear() override {}
    size_t get_num_layers() const override { return 1; }
    size_t get_num_cache_tensors() const override { return 1; }
    size_t get_block_size() const override { return 16; }
    std::string get_device() const override { return "CPU"; }
    size_t get_block_size_in_bytes() const override { return m_block_size_in_bytes; }
    size_t get_num_allocated_blocks() const override { return m_blocks.size(); }

    void read_block_to_host(size_t block_index, uint8_t* host_dst) const override {
        const auto& block = m_blocks.at(block_index);
        std::copy(block.begin(), block.end(), host_dst);
    }

    void write_block_from_host(size_t block_index, const uint8_t* host_src) override {
        auto& block = m_blocks.at(block_index);
        std::copy(host_src, host_src + block.size(), block.begin());
    }

    // Test helpers -----------------------------------------------------------
    void fill_block(size_t block_index, uint8_t value) {
        auto& block = m_blocks.at(block_index);
        std::fill(block.begin(), block.end(), value);
    }

    bool block_equals(size_t block_index, uint8_t value) const {
        const auto& block = m_blocks.at(block_index);
        return std::all_of(block.begin(), block.end(), [value](uint8_t b) { return b == value; });
    }

private:
    size_t m_block_size_in_bytes;
    std::vector<std::vector<uint8_t>> m_blocks;
};

constexpr size_t BLOCK_BYTES = 64;

OffloadConfig make_config(size_t max_bytes = 0) {
    OffloadConfig config;
    config.enabled = true;
    config.tier = OffloadConfig::Tier::HOST_RAM;
    config.max_offload_bytes = max_bytes;
    return config;
}

}  // namespace

TEST(OffloadManagerTest, DisabledByDefaultDoesNothing) {
    OffloadConfig config;  // enabled == false
    FakeCacheManager cache(4, BLOCK_BYTES);
    OffloadManager manager(config, BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    EXPECT_FALSE(manager.enabled());
    EXPECT_FALSE(manager.offload(42, {0}));
    EXPECT_FALSE(manager.is_offloaded(42));
}

TEST(OffloadManagerTest, OffloadThenRestoreRoundTrip) {
    FakeCacheManager cache(4, BLOCK_BYTES);
    OffloadManager manager(make_config(), BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    cache.fill_block(0, 0xAB);
    ASSERT_TRUE(manager.offload(/*hash=*/7, {0}));
    EXPECT_TRUE(manager.is_offloaded(7));
    EXPECT_EQ(manager.num_offloaded(), 1u);
    EXPECT_EQ(manager.bytes_used(), BLOCK_BYTES);

    // Simulate the device block being reused/overwritten by another sequence.
    cache.fill_block(0, 0x00);
    EXPECT_TRUE(cache.block_equals(0, 0x00));

    // Restore into a (different) free device block and verify the original contents come back.
    ASSERT_TRUE(manager.restore(/*hash=*/7, {1}));
    EXPECT_TRUE(cache.block_equals(1, 0xAB));
    EXPECT_FALSE(manager.is_offloaded(7));
    EXPECT_EQ(manager.num_offloaded(), 0u);
    EXPECT_EQ(manager.bytes_used(), 0u);
}

TEST(OffloadManagerTest, ReoffloadingSameHashIsNoOp) {
    FakeCacheManager cache(4, BLOCK_BYTES);
    OffloadManager manager(make_config(), BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    cache.fill_block(0, 0x11);
    ASSERT_TRUE(manager.offload(1, {0}));
    // Second offload of the same hash must not grow memory or copy again.
    cache.fill_block(0, 0x22);
    ASSERT_TRUE(manager.offload(1, {0}));
    EXPECT_EQ(manager.num_offloaded(), 1u);
    EXPECT_EQ(manager.bytes_used(), BLOCK_BYTES);

    ASSERT_TRUE(manager.restore(1, {2}));
    // The originally stored contents (0x11) must be preserved, not the later 0x22.
    EXPECT_TRUE(cache.block_equals(2, 0x11));
}

TEST(OffloadManagerTest, EvictsLeastRecentlyUsedWhenOverBudget) {
    FakeCacheManager cache(8, BLOCK_BYTES);
    // Budget for exactly two blocks.
    OffloadManager manager(make_config(/*max_bytes=*/2 * BLOCK_BYTES), BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    cache.fill_block(0, 0xA0);
    cache.fill_block(1, 0xB0);
    cache.fill_block(2, 0xC0);

    ASSERT_TRUE(manager.offload(10, {0}));
    ASSERT_TRUE(manager.offload(11, {1}));
    EXPECT_EQ(manager.num_offloaded(), 2u);

    // Touch hash 10 so that 11 becomes the least recently used entry.
    EXPECT_TRUE(manager.is_offloaded(10));

    // Offloading a third block must evict the LRU entry (hash 10, never touched after insertion).
    ASSERT_TRUE(manager.offload(12, {2}));
    EXPECT_EQ(manager.num_offloaded(), 2u);
    EXPECT_FALSE(manager.is_offloaded(10));
    EXPECT_TRUE(manager.is_offloaded(11));
    EXPECT_TRUE(manager.is_offloaded(12));
    EXPECT_EQ(manager.bytes_used(), 2 * BLOCK_BYTES);
}

TEST(OffloadManagerTest, RestoreMissingHashReturnsFalse) {
    FakeCacheManager cache(4, BLOCK_BYTES);
    OffloadManager manager(make_config(), BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    EXPECT_FALSE(manager.restore(/*hash=*/999, {0}));
}

TEST(OffloadManagerTest, DropAndClearReleaseMemory) {
    FakeCacheManager cache(4, BLOCK_BYTES);
    OffloadManager manager(make_config(), BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    ASSERT_TRUE(manager.offload(1, {0}));
    ASSERT_TRUE(manager.offload(2, {1}));
    EXPECT_EQ(manager.num_offloaded(), 2u);

    manager.drop(1);
    EXPECT_FALSE(manager.is_offloaded(1));
    EXPECT_EQ(manager.num_offloaded(), 1u);
    EXPECT_EQ(manager.bytes_used(), BLOCK_BYTES);

    manager.clear();
    EXPECT_EQ(manager.num_offloaded(), 0u);
    EXPECT_EQ(manager.bytes_used(), 0u);
}

TEST(OffloadManagerTest, MultiBlockSetRoundTrip) {
    FakeCacheManager cache(8, BLOCK_BYTES);
    OffloadManager manager(make_config(), BLOCK_BYTES);
    manager.set_cache_manager(&cache);

    cache.fill_block(0, 0x01);
    cache.fill_block(1, 0x02);
    ASSERT_TRUE(manager.offload(/*hash=*/5, {0, 1}));
    EXPECT_EQ(manager.bytes_used(), 2 * BLOCK_BYTES);

    ASSERT_TRUE(manager.restore(/*hash=*/5, {4, 5}));
    EXPECT_TRUE(cache.block_equals(4, 0x01));
    EXPECT_TRUE(cache.block_equals(5, 0x02));
}
