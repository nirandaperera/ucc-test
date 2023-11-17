//
// Created by niranda on 11/1/23.
//
#pragma once

#include <memory>
#include <chrono>
#include <csignal>
#include <deque>
#include <thread>
#include <atomic>
#include "data.hpp"
#include "third-party/CLI11.hpp"

namespace ucc_test {

class Benchmark {
 public:
  Benchmark(MPI_Comm comm, std::vector<std::string> *bench_args) : rank(get_mpi_rank(comm)),
                                                                   world_size(get_mpi_world_size(comm)),
                                                                   iter(-1) {
    CLI::App app;
    app.allow_extras();
    app.add_option("--iter", iter)->required();
    app.parse(*bench_args);

    std::cout << "rank " << rank << " sz " << world_size << " iter " << iter << std::endl;
  }

  virtual ~Benchmark() = default;

  virtual ucc_status_t Run() = 0;

 protected:
  uint32_t rank, world_size, iter;
};

class CtxCreateBenchmark : public Benchmark {
 public:
  CtxCreateBenchmark(MPI_Comm comm, std::vector<std::string> *bench_args) : Benchmark(comm, bench_args) {
    kTimeMap.clear();
    kTimeMap.reserve(10 * iter);
  }

  static std::string_view Name() { return "ctx_create"; }

  ucc_status_t Run() override {
    std::array<double, 5> t{};

    ucc_lib_h lib;
    UPDATE_TIMING(t[0], init_ucc(&lib));

    for (int i = 0; i < iter; i++) {
      ucc_context_h ucc_ctx;
      UPDATE_TIMING(t[1], create_ucc_ctx(lib, rank, world_size, &ucc_ctx));

      ucc_team_h ucc_team;
      UPDATE_TIMING(t[2], create_ucc_team(ucc_ctx, rank, world_size, &ucc_team));

      UPDATE_TIMING(t[3], destroy_ucc_team(ucc_team));

      UPDATE_TIMING(t[4], ucc_context_destroy(ucc_ctx));
    }
    std::cout << "TIMINGS " << rank << ","
              << t[0] << ","
              << t[1] / iter << ","
              << t[2] / iter << ","
              << t[3] / iter << ","
              << t[4] / iter << std::endl;
    return UCC_OK;
  }
};

class AllGatherBenchmark : public Benchmark {
 public:
  AllGatherBenchmark(MPI_Comm comm, std::vector<std::string> *bench_args) : Benchmark(comm, bench_args) {
    CLI::App app;
    app.add_option("-t", num_tables)->default_val(1);
    app.add_option("--min_s", min_table_sz)->required()->transform(CLI::AsSizeValue(/*kb_is_1000=*/false));
    app.add_option("--max_s", max_table_sz)->required()->transform(CLI::AsSizeValue(/*kb_is_1000=*/false));
    app.add_option("--buf_s", buf_sz)->required()->transform(CLI::AsSizeValue(/*kb_is_1000=*/false));
    app.parse(*bench_args);

    if (!buf_sz || !min_table_sz || !max_table_sz) {
      throw std::runtime_error("malformed buffer/ table size");
    }

    buf_sz = round_up(buf_sz, sizeof(uint32_t)); // round up buffer sizes to sizeof(int)

    std::cout << "t sz " << min_table_sz << "," << max_table_sz << " buf " << buf_sz << " num t " << num_tables
              << std::endl;
  }

  static std::string_view Name() { return "allgather"; }

  ucc_status_t InitUcc() {
    CHECK_UCC_OK(init_ucc(&lib))
    CHECK_UCC_OK(create_ucc_ctx(lib, rank, world_size, &ucc_ctx))
    return create_ucc_team(ucc_ctx, rank, world_size, &ucc_team);
  }

  ucc_status_t AllGatherNumBuffers(const Table &table, std::vector<uint32_t> *num_buffers) {
    ucc_coll_args_t args;
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;

    uint32_t src = table.buffers.size();
    num_buffers->resize(world_size);

    args.src.info = {.buffer = &src, .count = 1, .datatype = UCC_DT_UINT32, .mem_type = UCC_MEMORY_TYPE_HOST};
    args.dst.info =
        {.buffer = num_buffers->data(), .count = world_size, .datatype = UCC_DT_UINT32, .mem_type = UCC_MEMORY_TYPE_HOST};

    ucc_coll_req_h req;
    CHECK_UCC_OK(ucc_collective_init(&args, &req, ucc_team))

    CHECK_UCC_OK(ucc_collective_post(req))

    ucc_status_t status;
    while (UCC_INPROGRESS == (status = ucc_collective_test(req))) {
      ucc_context_progress(ucc_ctx);
    }
    CHECK_UCC_OK(status);

    return ucc_collective_finalize(req);
  }

  ucc_status_t AllGatherBuffer(uint8_t *snd_buf, uint8_t *rec_buf, ucc_coll_req_h *req) {
    ucc_coll_args_t args;
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;

    args.src.info = {.buffer = snd_buf, .count = buf_sz, .datatype = UCC_DT_UINT8, .mem_type = UCC_MEMORY_TYPE_HOST};
    args.dst.info =
        {.buffer = rec_buf, .count = world_size * buf_sz, .datatype = UCC_DT_UINT8, .mem_type = UCC_MEMORY_TYPE_HOST};

    CHECK_UCC_OK(ucc_collective_init(&args, req, ucc_team))

    return ucc_collective_post(*req);
  }

  ucc_status_t AllGatherVBuffer(uint8_t *snd_buf,
                                size_t snd_cnt,
                                uint8_t *rec_buf,
                                uint32_t *rec_counts,
                                uint32_t *displacements,
                                ucc_coll_req_h *req) {
    ucc_coll_args_t args;
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

    args.src.info = {.buffer = snd_buf, .count = snd_cnt, .datatype = UCC_DT_UINT8, .mem_type = UCC_MEMORY_TYPE_HOST};
    args.dst.info_v =
        {.buffer = rec_buf, .counts = (ucc_count_t *) rec_counts, .displacements=(ucc_aint_t *) displacements,
            .datatype = UCC_DT_UINT8, .mem_type = UCC_MEMORY_TYPE_HOST};

    CHECK_UCC_OK(ucc_collective_init(&args, req, ucc_team))

    return ucc_collective_post(*req);
  }

  ucc_status_t CreateAllGatherBufferRequests(const Table &table,
                                             const std::vector<uint32_t> &num_buffers,
                                             uint32_t tot_num_buf,
                                             uint32_t min_num_buf,
                                             uint32_t max_num_buf,
                                             Buffer *rec_buffer) {
    rec_buffer->Resize(tot_num_buf * buf_sz);

    auto &table_bufs = const_cast<std::vector<Buffer> &>(table.buffers);

    size_t idx = 0;
    // first do allgather where every table has a buffer
    for (; idx < min_num_buf; idx++) {
      assert(buf_sz == table_bufs[idx].Size());
      ucc_coll_req_h req;
      uint8_t *rec_pos = rec_buffer->Data() + idx * world_size * buf_sz;
      CHECK_UCC_OK(AllGatherBuffer(table_bufs[idx].Data(), rec_pos, &req))
      pending_requests.emplace_back(req);
    }

    uint8_t *rec_pos = rec_buffer->Data() + idx * world_size * buf_sz;
    // Then do allgatherv with empty buffers when there are no beffers to offer
    for (; idx < max_num_buf; idx++) {

      uint32_t snd_count = 0;
      uint8_t *snd_buf = nullptr;
      if (idx < table_bufs.size()) {
        snd_count = buf_sz;
        snd_buf = table_bufs[idx].Data();
      }

      std::vector<uint32_t> rec_counts(world_size, 0);
      std::transform(num_buffers.cbegin(), num_buffers.cend(), rec_counts.begin(),
                     [buf_sz_ = buf_sz, idx](uint32_t num_buf) {
                       return (idx < num_buf) * buf_sz_;
                     });

      std::vector<uint32_t> displacements(world_size, 0);
      std::partial_sum(rec_counts.begin(), rec_counts.end() - 1, displacements.begin() + 1);

//      print_array("counts ", rank, rec_counts);
//      print_array("disp ", rank, displacements);

      ucc_coll_req_h req;
      CHECK_UCC_OK(AllGatherVBuffer(snd_buf, snd_count, rec_pos, rec_counts.data(), displacements.data(), &req))

      rec_pos += std::accumulate(rec_counts.cbegin(), rec_counts.cend(), uint32_t(0));
      pending_requests.emplace_back(req, std::move(rec_counts), std::move(displacements));
    }
    return UCC_OK;
  }

  ucc_status_t ProgressRequests() {
    // progress the requests
    while (!pending_requests.empty()) {
      // every iteration progress context
      ucc_context_progress(ucc_ctx);

      auto &req = pending_requests.front();
      auto status = ucc_collective_test(req.req);
      if (status == UCC_OK) {
        // request completed
        ucc_collective_finalize(req.req);
        pending_requests.pop_front();
      } else if (status < 0) {
        return status; // an error has occurred
      }
    }
    return UCC_OK;
  }

  ucc_status_t Run() override {
    tables = generate_tables(num_tables, min_table_sz, max_table_sz, buf_sz, rank, world_size);
//    PrintTables();

    // init ucc
    CHECK_UCC_OK(InitUcc())

    std::array<double, 3> t{};

    uint32_t min_num_buf, max_num_buf, tot_num_buf;
    for (uint32_t i = 0; i < iter; i++) {
      std::vector<uint32_t> num_buffers;

      UPDATE_TIMING(t[0], AllGatherNumBuffers(tables[0], &num_buffers));
      auto min_max = std::minmax_element(num_buffers.begin(), num_buffers.end());
      min_num_buf = *min_max.first;
      max_num_buf = *min_max.second;
      tot_num_buf = std::accumulate(num_buffers.begin(), num_buffers.end(), uint32_t(0));
//    print_array("num buf\t", rank, num_buffers);

      Buffer rec_buffer;
      UPDATE_TIMING(t[1], CreateAllGatherBufferRequests(tables[0],
                                                        num_buffers,
                                                        tot_num_buf,
                                                        min_num_buf,
                                                        max_num_buf,
                                                        &rec_buffer));

      UPDATE_TIMING(t[2], ProgressRequests());

//      PrintOutput(rec_buffer);
    }
    std::cout << "TIMINGS(" << iter << ") " << rank << "," << tot_num_buf * buf_sz << "," << t[0] / iter << ","
              << t[1] / iter << "," << t[2] / iter << std::endl;

    return UCC_OK;
  }

 private:
  struct Request {
    explicit Request(ucc_coll_req_h req) : req(req) {}
    Request(ucc_coll_req_h req, std::vector<uint32_t> &&counts, std::vector<uint32_t> &&displacements)
        : req(req), counts(std::move(counts)), displacements(std::move(displacements)) {}
    ucc_coll_req_h req;
    // for allgatherv requests
    std::vector<uint32_t> counts{};
    std::vector<uint32_t> displacements{};
  };

  void PrintTables() {
    for (size_t i = 0; i < tables.size(); i++) {
      std::cout << rank << " " << i << " " << tables[i] << std::endl << std::flush;
    }
  }

  void PrintOutput(Buffer &rec_buffer) {
    uint8_t *p = rec_buffer.Data();
    std::cout << rank << " OUT:[";
    for (size_t i = 0; i < rec_buffer.Size(); i += buf_sz) {
      std::cout << *reinterpret_cast<uint32_t *>(p + i) << ",";
    }
    std::cout << "]" << std::endl;
  }

  size_t num_tables{}, min_table_sz{}, max_table_sz{}, buf_sz{};

  // ucc
  ucc_lib_h lib{};
  ucc_context_h ucc_ctx{};
  ucc_team_h ucc_team{};

  // data
  std::vector<Table> tables;

  std::deque<Request> pending_requests{};
};

std::unique_ptr<Benchmark> create_bench(const std::string_view &name, MPI_Comm comm,
                                        std::vector<std::string> *bench_args) {
  if (name == CtxCreateBenchmark::Name()) {
    return std::make_unique<CtxCreateBenchmark>(comm, bench_args);
  } else if (name == AllGatherBenchmark::Name()) {
    return std::make_unique<AllGatherBenchmark>(comm, bench_args);
  }

  return nullptr;
}

} // namespace ucc_test4