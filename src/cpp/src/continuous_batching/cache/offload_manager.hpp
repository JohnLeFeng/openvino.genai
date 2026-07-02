// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "openvino/genai/scheduler_config.hpp"
#include "continuous_batching/cache/i_cache_manager.hpp"

namespace ov::genai {

/**
 * @brief Stores prefix-cached KV blocks in a cheaper memory tier (host RAM, and later disk) so they can be
 * restored on a future prefix-cache hit instead of being recomputed.
 *
 * The manager is storage-only: it never touches the device tensors directly. All physical reads/writes are
 * delegated to an @ref ICacheManager via its host transfer primitives, keeping the manager device-agnostic.
 *
 * Entries are keyed by the prefix-cache content hash (the same chained hash used by the prefix cache), so a
 * restore is only valid when a new request reproduces the identical prefix.
 */
class OffloadManager {
public:
    /**
     * @param config Offload configuration (tier, budget, trigger threshold).
     * @param block_size_in_bytes Size in bytes of a single physical KV block across all decoder layers
     *        (key + value), as reported by ICacheManager::get_block_size_in_bytes().
     */
    OffloadManager(const OffloadConfig& config, size_t block_size_in_bytes);

    /// @brief Set the cache manager used for physical device<->host block transfers.
    void set_cache_manager(ICacheManager* cache) {
        m_cache = cache;
    }

    /// @return Whether offloading is enabled.
    bool enabled() const {
        return m_config.enabled;
    }

    /**
     * @brief Decide whether prefix blocks should be demoted to the offload tier given current device pressure.
     * @param num_free_device_blocks Number of free blocks currently available on the device.
     */
    bool should_offload(size_t num_free_device_blocks) const;

    /// @return Whether a block set for @p hash is currently held in the offload tier.
    bool is_offloaded(uint64_t hash) const;

    /// @return Number of block sets currently held in the offload tier.
    size_t num_offloaded() const {
        return m_storage.size();
    }

    /// @return Total number of bytes currently held in the offload tier.
    size_t bytes_used() const {
        return m_bytes_used;
    }

    /**
     * @brief Copy KV data for the given physical block indices out of device memory into the tier under @p hash.
     * @param hash Prefix-cache content hash identifying the block set.
     * @param block_indices Physical block indices (one per block-table layer) to read from the device.
     * @param cache Cache manager used to read the device blocks into host memory.
     * @return True if the block set was stored; false if it could not fit within the configured budget
     *         (in which case nothing is stored and the caller should fall back to overwriting/discarding).
     */
    bool offload(uint64_t hash, const std::vector<size_t>& block_indices, const ICacheManager& cache);

    /// @brief Convenience overload using the cache manager set via @ref set_cache_manager.
    bool offload(uint64_t hash, const std::vector<size_t>& block_indices);

    /**
     * @brief Copy KV data stored under @p hash back into the given physical block indices, removing the entry.
     * @param hash Prefix-cache content hash identifying the block set.
     * @param block_indices Physical block indices (one per block-table layer) to write on the device.
     * @param cache Cache manager used to write the device blocks from host memory.
     * @return True if the block set was restored; false if @p hash is not present or the index count mismatches.
     */
    bool restore(uint64_t hash, const std::vector<size_t>& block_indices, ICacheManager& cache);

    /// @brief Convenience overload using the cache manager set via @ref set_cache_manager.
    bool restore(uint64_t hash, const std::vector<size_t>& block_indices);

    /// @brief Discard a stored entry without restoring it (tier eviction).
    void drop(uint64_t hash);

    /// @brief Remove all stored entries.
    void clear();

private:
    struct Entry {
        std::vector<uint8_t> data;  ///< Concatenation of block_indices.size() blocks.
        std::list<uint64_t>::iterator lru_it;  ///< Position in m_lru.
    };

    /// Evict least-recently-used entries until @p additional_bytes can be admitted. Returns false if impossible.
    bool ensure_budget(size_t additional_bytes);

    /// Move @p hash to the most-recently-used end of the LRU list.
    void touch(uint64_t hash);

    /// Remove and discard the least-recently-used entry. Returns false if the tier is empty.
    bool evict_lru();

    OffloadConfig m_config;
    size_t m_block_size_in_bytes;
    size_t m_bytes_used = 0;
    ICacheManager* m_cache = nullptr;
    std::unordered_map<uint64_t, Entry> m_storage;
    std::list<uint64_t> m_lru;  ///< front = least recently used, back = most recently used.
};

}  // namespace ov::genai
