// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/offload_manager.hpp"

#include "openvino/core/except.hpp"

namespace ov::genai {

OffloadManager::OffloadManager(const OffloadConfig& config, size_t block_size_in_bytes)
    : m_config(config),
      m_block_size_in_bytes(block_size_in_bytes) {
    OPENVINO_ASSERT(block_size_in_bytes > 0, "OffloadManager requires a non-zero block size in bytes");
    // Disk tier is planned for a later milestone; only host RAM is implemented for now.
    OPENVINO_ASSERT(!m_config.enabled || m_config.tier == OffloadConfig::Tier::HOST_RAM,
                    "OffloadManager currently supports only the HOST_RAM tier");
}

bool OffloadManager::should_offload(size_t num_free_device_blocks) const {
    if (!m_config.enabled) {
        return false;
    }
    // When trigger_free_blocks is 0, demotion is driven purely by the caller (e.g. on cache exhaustion).
    return num_free_device_blocks <= m_config.trigger_free_blocks;
}

bool OffloadManager::is_offloaded(uint64_t hash) const {
    return m_storage.find(hash) != m_storage.end();
}

bool OffloadManager::offload(uint64_t hash, const std::vector<size_t>& block_indices, const ICacheManager& cache) {
    if (!m_config.enabled || block_indices.empty()) {
        return false;
    }
    // Re-offloading an already-stored hash is a no-op (keep the existing copy, refresh recency).
    if (is_offloaded(hash)) {
        touch(hash);
        return true;
    }

    const size_t required_bytes = block_indices.size() * m_block_size_in_bytes;
    if (!ensure_budget(required_bytes)) {
        return false;
    }

    Entry entry;
    entry.data.resize(required_bytes);
    for (size_t i = 0; i < block_indices.size(); ++i) {
        cache.read_block_to_host(block_indices[i], entry.data.data() + i * m_block_size_in_bytes);
    }

    m_lru.push_back(hash);
    entry.lru_it = std::prev(m_lru.end());
    m_bytes_used += required_bytes;
    m_storage.emplace(hash, std::move(entry));
    return true;
}

bool OffloadManager::offload(uint64_t hash, const std::vector<size_t>& block_indices) {
    OPENVINO_ASSERT(m_cache != nullptr, "OffloadManager::offload called before a cache manager was set");
    return offload(hash, block_indices, *m_cache);
}

bool OffloadManager::restore(uint64_t hash, const std::vector<size_t>& block_indices, ICacheManager& cache) {
    auto it = m_storage.find(hash);
    if (it == m_storage.end()) {
        return false;
    }
    Entry& entry = it->second;
    const size_t required_bytes = block_indices.size() * m_block_size_in_bytes;
    OPENVINO_ASSERT(entry.data.size() == required_bytes,
                    "OffloadManager::restore block index count does not match the stored block set for hash ", hash);

    for (size_t i = 0; i < block_indices.size(); ++i) {
        cache.write_block_from_host(block_indices[i], entry.data.data() + i * m_block_size_in_bytes);
    }

    m_bytes_used -= entry.data.size();
    m_lru.erase(entry.lru_it);
    m_storage.erase(it);
    return true;
}

bool OffloadManager::restore(uint64_t hash, const std::vector<size_t>& block_indices) {
    OPENVINO_ASSERT(m_cache != nullptr, "OffloadManager::restore called before a cache manager was set");
    return restore(hash, block_indices, *m_cache);
}

void OffloadManager::drop(uint64_t hash) {
    auto it = m_storage.find(hash);
    if (it == m_storage.end()) {
        return;
    }
    m_bytes_used -= it->second.data.size();
    m_lru.erase(it->second.lru_it);
    m_storage.erase(it);
}

void OffloadManager::clear() {
    m_storage.clear();
    m_lru.clear();
    m_bytes_used = 0;
}

void OffloadManager::touch(uint64_t hash) {
    auto it = m_storage.find(hash);
    if (it == m_storage.end()) {
        return;
    }
    m_lru.erase(it->second.lru_it);
    m_lru.push_back(hash);
    it->second.lru_it = std::prev(m_lru.end());
}

bool OffloadManager::evict_lru() {
    if (m_lru.empty()) {
        return false;
    }
    const uint64_t lru_hash = m_lru.front();
    drop(lru_hash);
    return true;
}

bool OffloadManager::ensure_budget(size_t additional_bytes) {
    // 0 means unbounded for the host RAM tier.
    if (m_config.max_offload_bytes == 0) {
        return true;
    }
    if (additional_bytes > m_config.max_offload_bytes) {
        // A single block set cannot fit even in an empty tier.
        return false;
    }
    while (m_bytes_used + additional_bytes > m_config.max_offload_bytes) {
        if (!evict_lru()) {
            return false;
        }
    }
    return true;
}

}  // namespace ov::genai
