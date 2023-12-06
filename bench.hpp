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
#include <iomanip>
#include <ostream>
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

  struct timing {
    explicit timing(uint32_t iter) : iter(iter) {}
    uint32_t iter;
    double init{}, c_create{}, t_create{}, barrier{}, t_destroy{}, c_destroy{};
    friend std::ostream &operator<<(std::ostream &os, const timing &t) {
      os << t.init << ","
         << t.c_create / t.iter << ","
         << t.t_create / t.iter << ","
         << t.barrier / t.iter << ","
         << t.t_destroy / t.iter << ","
         << t.c_destroy / t.iter;
      return os;
    }
  };

  ucc_status_t Run() override {
    timing t(iter);
    ucc_lib_h lib;
    UPDATE_TIMING(t.init, init_ucc(&lib));

    for (int i = 0; i < iter; i++) {
      ucc_context_h ucc_ctx;
      UPDATE_TIMING(t.c_create, create_ucc_ctx(lib, rank, world_size, MPI_COMM_WORLD, &ucc_ctx));

      ucc_team_h ucc_team;
      UPDATE_TIMING(t.t_create, create_ucc_team(ucc_ctx, rank, world_size, MPI_COMM_WORLD, &ucc_team));

      UPDATE_TIMING(t.barrier, ucc_barrier(ucc_ctx, ucc_team));

      UPDATE_TIMING(t.t_destroy, destroy_ucc_team(ucc_team));

      UPDATE_TIMING(t.c_destroy, ucc_context_destroy(ucc_ctx));
    }
    std::cout << "TIMINGS " << rank << "," << t << std::endl;
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

    std::cout << "t_sz " << min_table_sz << "," << max_table_sz << " buf " << buf_sz << " num_t " << num_tables
              << std::endl;
  }

  static std::string_view Name() { return "allgather"; }

  ucc_status_t InitUcc() {
    tables = generate_tables(num_tables, min_table_sz, max_table_sz, buf_sz, rank, world_size);
    rec_buffer.Resize(world_size * max_table_sz); // allocate memory for max size
    num_buffers.resize(world_size, 0);

    size_t tot_buffers = std::accumulate(tables.begin(), tables.end(), 0, [](size_t sum, const Table &t) {
      return sum + t.buffers.size();
    });
    mem_regions.reserve(tot_buffers + 2); // +2 for rec_buffer and num_buffer
    for (auto &t : tables) {
      for (auto &b : t.buffers) {
        mem_regions.emplace_back(ucc_mem_map_t{.address=b.Data(), .len=b.Size()});
      }
    }
    mem_regions.emplace_back(ucc_mem_map_t{.address=rec_buffer.Data(), .len=rec_buffer.Size()});
    mem_regions.emplace_back(ucc_mem_map_t{.address=num_buffers.data(), .len=num_buffers.size() * sizeof(uint32_t)});

    // ucc lib
    CHECK_UCC_OK(init_ucc(&lib))

    // ucc ctx
    ucc_context_config_h ctx_config;
    CHECK_UCC_OK(ucc_context_config_read(lib, /*filename=*/nullptr, &ctx_config))

    ucc_context_params_t ctx_params;

    ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_TYPE;
    ctx_params.type = UCC_CONTEXT_EXCLUSIVE;

    ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_SYNC_TYPE;
    ctx_params.sync_type = UCC_NO_SYNC_COLLECTIVES;

    ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob =
        {.allgather= oob_allgather<ctx_type>, .req_test = oob_allgather_test, .req_free = oob_allgather_free,
            .coll_info = MPI_COMM_WORLD, .n_oob_eps = world_size, .oob_ep = rank};

    ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    ctx_params.mem_params = {mem_regions.data(), mem_regions.size()};

    CHECK_UCC_OK(ucc_context_create(lib, &ctx_params, ctx_config, &ucc_ctx))
    ucc_context_config_release(ctx_config);

    return create_ucc_team(ucc_ctx, rank, world_size, MPI_COMM_WORLD, &ucc_team);
  }

  ucc_status_t DestroyUcc() {
    CHECK_UCC_OK(destroy_ucc_team(ucc_team))
    return ucc_context_destroy(ucc_ctx);
  }

  ucc_status_t AllGatherNumBuffers(const Table &table) {
    ucc_coll_args_t args;
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;

    uint32_t src = table.buffers.size();

    args.src.info = {.buffer = &src, .count = 1, .datatype = UCC_DT_UINT32, .mem_type = UCC_MEMORY_TYPE_HOST};
    args.dst.info =
        {.buffer = num_buffers.data(), .count = world_size, .datatype = UCC_DT_UINT32, .mem_type = UCC_MEMORY_TYPE_HOST};

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
                                             uint32_t tot_num_buf,
                                             uint32_t min_num_buf,
                                             uint32_t max_num_buf) {

    auto &table_bufs = const_cast<std::vector<Buffer> &>(table.buffers);

    size_t idx = 0;
    // first do allgather where every table has a buffer
    for (; idx < min_num_buf; idx++) {
      assert(buf_sz == table_bufs[idx].Size());
      ucc_coll_req_h req;
      uint8_t *rec_pos = rec_buffer.Data() + idx * world_size * buf_sz;
      CHECK_UCC_OK(AllGatherBuffer(table_bufs[idx].Data(), rec_pos, &req))
      pending_requests.emplace_back(req);
    }

    uint8_t *rec_pos = rec_buffer.Data() + idx * world_size * buf_sz;
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

      // go through all requests and finalize finished ones
      for (auto i = pending_requests.begin(); i < pending_requests.end(); i++) {
        auto status = ucc_collective_test((*i).req);
        if (status == UCC_OK) {
          // request completed
          ucc_collective_finalize((*i).req);
          pending_requests.erase(i);
        } else if (status < 0) {
          return status; // an error has occurred
        }
      }
    }
    return UCC_OK;
  }

  ucc_status_t Run() override {
//    PrintTables();

    // init ucc
    CHECK_UCC_OK(InitUcc())
    CHECK_UCC_OK(ucc_barrier(ucc_ctx, ucc_team)) // barrier after init

//    std::array<double, 5> t{};

    std::vector<double> t(5 * iter, 0);
    uint32_t min_num_buf, max_num_buf, tot_num_buf;
    for (uint32_t i = 0; i < iter; i++) {
      auto start = std::chrono::high_resolution_clock::now();

      UPDATE_TIMING(t[i * 5 + 0], AllGatherNumBuffers(tables[0]));
      auto min_max = std::minmax_element(num_buffers.begin(), num_buffers.end());
      min_num_buf = *min_max.first;
      max_num_buf = *min_max.second;
      tot_num_buf = std::accumulate(num_buffers.begin(), num_buffers.end(), uint32_t(0));
//    print_array("num buf\t", rank, num_buffers);

      UPDATE_TIMING(t[i * 5 + 1], CreateAllGatherBufferRequests(tables[0],
                                                                tot_num_buf,
                                                                min_num_buf,
                                                                max_num_buf));

      UPDATE_TIMING(t[i * 5 + 2], ProgressRequests());

      auto end = std::chrono::high_resolution_clock::now();
//      PrintOutput();
      t[i * 5 + 3] += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

      UPDATE_TIMING(t[i * 5 + 4], ucc_barrier(ucc_ctx, ucc_team));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(rank * 50));
    std::stringstream ss;
    for (size_t i = 0; i < iter * 5; i += 5) {
      ss << std::fixed << std::setprecision(4) << world_size
                                               << " TIMINGS(" << iter << ") " << rank << "\t"
                                               << buf_sz << "\t" << tot_num_buf * buf_sz << "\t"
                                               << t[i + 0] << "\t" << t[i + 1] << "\t" << t[i + 2] << "\t" << t[i + 3]
                                               << "\t" << t[i + 4] << "\t"
                                               << i / 5 << std::endl;
    }
    std::cout << ss.rdbuf();

    return DestroyUcc();
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

  void PrintOutput() {
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
  std::vector<uint32_t> num_buffers;
  Buffer rec_buffer;

  std::deque<Request> pending_requests{};
  std::vector<ucc_mem_map_t> mem_regions{};
};

class MultiCtxTest : public Benchmark {
 public:
  MultiCtxTest(MPI_Comm comm, std::vector<std::string> *bench_args) : Benchmark(comm, bench_args) {
    CLI::App app;
    app.add_option("--ctx_sz", ctx_sz)->required();
    app.add_option("--buf_sz", buf_sz)->default_val(/*1MB*/1 << 20)->transform(CLI::AsSizeValue(/*kb_is_1000=*/false));
    app.parse(*bench_args);

    if (ctx_sz >= world_size) {
      throw std::runtime_error("ctx_sz should be <World size");
    }

    // color 0 if rank < ctx_sz else 1
    if (rank < ctx_sz) {
      sub_comm_id = 0;
      sub_comm_rank = rank;
      sub_comm_world_sz = ctx_sz;
    } else {
      sub_comm_id = 1;
      sub_comm_rank = rank - ctx_sz;
      sub_comm_world_sz = world_size - ctx_sz;
    }

    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, sub_comm_id, sub_comm_rank, &sub_comm));
    if (sub_comm_rank != get_mpi_rank(sub_comm)) {
      throw std::runtime_error("sub comm rank mismatch");
    }

    if (sub_comm_world_sz != get_mpi_world_size(sub_comm)) {
      throw std::runtime_error("sub comm world size mismatch");
    }

    std::cout << "sub comm rank " << sub_comm_rank << " sz " << sub_comm_world_sz << std::endl;
  }

  static std::string_view Name() { return "multi_ctx"; }

  [[nodiscard]] ucc_status_t AllGatherBuffer(const ucc_context_h &ctx, const ucc_team_h &team, size_t w_sz) const {
    size_t num_elems = (buf_sz + sizeof(int64_t) - 1) / sizeof(int64_t);
    std::vector<int64_t> src(num_elems);
    std::iota(src.begin(), src.end(), 0);

    std::vector<int64_t> dest(num_elems * w_sz);

    ucc_coll_args_t args;
    args.mask = 0;
    args.coll_type = UCC_COLL_TYPE_ALLGATHER;

    args.src.info =
        {.buffer = src.data(), .count = num_elems, .datatype = UCC_DT_INT64, .mem_type = UCC_MEMORY_TYPE_HOST};
    args.dst.info =
        {.buffer = dest.data(), .count = num_elems * w_sz, .datatype = UCC_DT_INT64, .mem_type = UCC_MEMORY_TYPE_HOST};

    ucc_coll_req_h req;
    CHECK_UCC_OK(ucc_collective_init(&args, &req, team))

    CHECK_UCC_OK(ucc_collective_post(req));

    ucc_status_t status;
    while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
      ucc_context_progress(ctx);
    }
    CHECK_UCC_OK(status)

    CHECK_UCC_OK(ucc_collective_finalize(req))

    return check_all_gather_buffer(src, dest, w_sz);
  }

  ucc_status_t Execute(ucc_context_h &ctx, ucc_team_h &team, size_t r, size_t w_sz, MPI_Comm comm) {
    CHECK_UCC_OK(create_ucc_ctx(lib, r, w_sz, comm, &ctx))

    CHECK_UCC_OK(create_ucc_team(ctx, r, w_sz, comm, &team))

    CHECK_UCC_OK(ucc_barrier(ctx, team))

    CHECK_UCC_OK(AllGatherBuffer(ctx, team, w_sz))

    CHECK_UCC_OK(ucc_team_destroy(team))

    CHECK_UCC_OK(ucc_context_destroy(ctx))

    return UCC_OK;
  }

  ucc_status_t Run() override {
    CHECK_UCC_OK(init_ucc(&lib))

    std::thread sub_comm_thread([&]() {
      CHECK_UCC_OK(Execute(ctxs[1], teams[1], sub_comm_rank, sub_comm_world_sz, sub_comm))
      return UCC_OK;
    });

    CHECK_UCC_OK(Execute(ctxs[0], teams[0], rank, world_size, MPI_COMM_WORLD));

    sub_comm_thread.join();
    return UCC_OK;
  }

 private:
  int ctx_sz{}, buf_sz{}, sub_comm_id{}, sub_comm_rank{}, sub_comm_world_sz{};
  MPI_Comm sub_comm{};

  // ucc
  ucc_lib_h lib{};
  std::array<ucc_context_h, 2> ctxs{};
  std::array<ucc_team_h, 2> teams{};
};

std::unique_ptr<Benchmark> create_bench(const std::string_view &name, MPI_Comm comm,
                                        std::vector<std::string> *bench_args) {
  if (name == CtxCreateBenchmark::Name()) {
    return std::make_unique<CtxCreateBenchmark>(comm, bench_args);
  } else if (name == AllGatherBenchmark::Name()) {
    return std::make_unique<AllGatherBenchmark>(comm, bench_args);
  } else if (name == MultiCtxTest::Name()) {
    return std::make_unique<MultiCtxTest>(comm, bench_args);
  }

  return nullptr;
}

} // namespace ucc_test4