//
// Created by niranda on 11/13/23.
//

#pragma once

#include <vector>
#include <cassert>
#include <ostream>
#include <random>
#include "utils.hpp"

namespace ucc_test {

struct Buffer {
  Buffer() = default;
  explicit Buffer(size_t size) : data(size) {}
  Buffer(size_t size, uint32_t val) : data(size) {
    if (size % sizeof(uint32_t) != 0) {
      throw std::runtime_error("buffer size cannot accommodate uint32 value");
    }
    auto *begin = reinterpret_cast<uint32_t *>(data.data());
    std::fill_n(begin, size / sizeof(uint32_t), val);
  }

  friend std::ostream &operator<<(std::ostream &os, const Buffer &buffer) {
    os << "Buffer(" << buffer.data.size() << "," << *reinterpret_cast<const uint32_t *>(buffer.data.data()) << ")";
    return os;
  }

  void Resize(size_t size) { data.resize(size, 0); }
  [[nodiscard]] size_t Size() const { return data.size(); }
  uint8_t *Data() { return data.data(); }

  std::vector<uint8_t> data;
};

struct Table {
  std::vector<Buffer> buffers;

  friend std::ostream &operator<<(std::ostream &os, const Table &table) {
    os << "Table(num_buf:" << table.buffers.size() << ",buffers:[";
    for (const auto &b : table.buffers) {
      os << b << ",\t";
    }
    os << "])";
    return os;
  }
};

std::vector<Table> generate_tables(size_t num_tables,
                                   size_t min_table_sz,
                                   size_t max_table_sz,
                                   size_t buf_sz,
                                   uint32_t rank,
                                   uint32_t world_sz) {
  std::vector<Table> tables;
  tables.reserve(num_tables);

  std::mt19937 rng(rank);
  std::uniform_int_distribution<size_t> gen(min_table_sz, max_table_sz); // uniform, unbiased

  for (size_t t = 0; t < num_tables; t++) {
    size_t table_sz = round_up(gen(rng), buf_sz);
    size_t num_buf = table_sz / buf_sz;

    std::vector<Buffer> buffers;
    buffers.reserve(num_buf);

    for (size_t b = 0; b < num_buf; b++) {
      buffers.emplace_back(/*size*/buf_sz, /*value=*/world_sz * t + rank);
    }

    tables.emplace_back(Table{std::move(buffers)});
  }
  return tables;
}

}
