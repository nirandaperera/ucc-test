//
// Created by niranda on 11/1/23.
//

#pragma once

#include <mpi.h>
#include <chrono>
#include <ucc/api/ucc.h>
#include <unordered_map>

namespace ucc_test {

static std::unordered_map<void *,
                          std::tuple<std::chrono::time_point<std::chrono::high_resolution_clock>,
                                     std::string,
                                     size_t,
                                     int>> kTimeMap;

#define CHECK_UCC_OK(expr) \
    if (const auto& r = (expr); r != UCC_OK) {                                              \
    std::cerr << "UCC error: " << r << " in " << __FILE__ << ":" << __LINE__ << std::endl;  \
    return r;                                                                               \
}

#define UPDATE_TIMING(t, expr) \
    do{               \
    auto start_ = std::chrono::high_resolution_clock::now(); \
    CHECK_UCC_OK(expr)             \
    auto end_ = std::chrono::high_resolution_clock::now();   \
    t += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_ - start_).count(); \
    } while(0)

int get_mpi_rank(MPI_Comm comm) {
  int x;
  MPI_Comm_rank(comm, &x);
  return x;
}

int get_mpi_world_size(MPI_Comm comm) {
  int x;
  MPI_Comm_size(comm, &x);
  return x;
}

template<typename T>
constexpr T round_up(T val, T base) {
  return base * ((val + base - 1) / base);
}

template<typename T>
void print_array(const std::string &prefix, uint32_t rank, const std::vector<T> &vec) {
  std::cout << prefix << rank << " [";
  for (auto &v : vec) {
    std::cout << v << ",";
  }
  std::cout << "]\n";
}

char ctx_type[] = "CTX ";
char team_type[] = "TEAM ";

template<const char *Type>
ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen, void *coll_info, void **req) {
  auto comm = static_cast<MPI_Comm>(coll_info);
  MPI_Request request;

  if (MPI_Iallgather(sbuf, (int) msglen, MPI_BYTE, rbuf, (int) msglen, MPI_BYTE, comm,
                     &request) != MPI_SUCCESS) {
    return UCC_ERR_NO_MESSAGE;
  }
  *req = request;

//  std::cout << Type << get_mpi_rank(comm) << " oob allgather " << msglen << " tag " << request << std::endl;
//  auto r = kTimeMap.emplace(request,
//                            std::tuple{std::chrono::high_resolution_clock::now(), Type, msglen, get_mpi_rank(comm)});
//  if (!r.second) {
//    throw std::runtime_error("aaaa");
//  }
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void *req) {
  auto request = static_cast<MPI_Request>(req);
  int completed;

  MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
//  auto end = std::chrono::high_resolution_clock::now();
//  if (completed) {
//    const auto &iter = kTimeMap.find(req);
//    auto start = std::get<0>(iter->second);
//    kTimeMap.erase(iter);
//    auto t = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
//    std::cout << "MPITIME " << t << " " << std::get<1>(iter->second) << " " << std::get<2>(iter->second)
//              << " " << std::get<3>(iter->second)
//              << " " << req << std::endl;
//  }
  return completed ? UCC_OK : UCC_INPROGRESS;
}

ucc_status_t oob_allgather_free(void */*req*/) {
//    auto request = static_cast<MPI_Request>(req);
//    return MPI_Request_free(&request) == MPI_SUCCESS? UCC_OK: UCC_ERR_NO_MESSAGE;
  return UCC_OK;
}

ucc_status_t init_ucc(ucc_lib_h *lib) {
  ucc_lib_config_h lib_config;

  // read ucc lib config
  ucc_lib_params_t lib_params = {.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE | UCC_LIB_PARAM_FIELD_SYNC_TYPE,
      .thread_mode = UCC_THREAD_MULTIPLE,
      .coll_types = {},
      .reduction_types = {},
      .sync_type = UCC_NO_SYNC_COLLECTIVES};

  CHECK_UCC_OK(ucc_lib_config_read(/*env_prefix=*/nullptr,/*filename=*/nullptr, &lib_config))

  // init ucc
  CHECK_UCC_OK(ucc_init(&lib_params, lib_config, lib))
  ucc_lib_config_release(lib_config); // this is no longer needed

  return UCC_OK;
}

ucc_status_t create_ucc_ctx(ucc_lib_h lib, int rank, int world_size, ucc_context_h *ucc_ctx) {
  // init ucc context
  ucc_context_params_t ctx_params;
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB | UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_SYNC_TYPE;

  ctx_params.type = UCC_CONTEXT_SHARED;
  ctx_params.sync_type = UCC_NO_SYNC_COLLECTIVES;

  ctx_params.oob.allgather = oob_allgather<ctx_type>;
  ctx_params.oob.req_test = oob_allgather_test;
  ctx_params.oob.req_free = oob_allgather_free;
  ctx_params.oob.coll_info = MPI_COMM_WORLD;
  ctx_params.oob.n_oob_eps = world_size;
  ctx_params.oob.oob_ep = rank;

  ucc_context_config_h ctx_config;
  CHECK_UCC_OK(ucc_context_config_read(lib, /*filename=*/nullptr, &ctx_config))

  CHECK_UCC_OK(ucc_context_create(lib, &ctx_params, ctx_config, ucc_ctx))
  ucc_context_config_release(ctx_config);

  return UCC_OK;
}

ucc_status_t create_ucc_team(ucc_context_h ucc_ctx, int rank, int world_size, ucc_team_h *ucc_team) {
  ucc_team_params_t team_params;

  team_params.mask = UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_ORDERING | UCC_TEAM_PARAM_FIELD_TEAM_SIZE |
      UCC_TEAM_PARAM_FIELD_SYNC_TYPE;

  team_params.oob.allgather = oob_allgather<team_type>;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = MPI_COMM_WORLD;
  team_params.oob.n_oob_eps = world_size;
  team_params.oob.oob_ep = rank;

  team_params.ordering = UCC_COLLECTIVE_INIT_AND_POST_UNORDERED;

  team_params.team_size = world_size;

  team_params.sync_type = UCC_NO_SYNC_COLLECTIVES;

  auto status = ucc_team_create_post(&ucc_ctx, /*num_contexts=*/1, &team_params, ucc_team);
  CHECK_UCC_OK(status)
  while (UCC_INPROGRESS == (status = ucc_team_create_test(*ucc_team))) {}

  return status;
}

ucc_status_t destroy_ucc_team(ucc_team_h &ucc_team) {
  ucc_status_t status;
  while (UCC_INPROGRESS == (status = ucc_team_destroy(ucc_team))) {}
  return status;
}

ucc_status_t ucc_barrier(ucc_context_h ctx, ucc_team_h team) {
  ucc_coll_args_t args_;
  ucc_coll_req_h req;

  args_.mask = 0;
  args_.coll_type = UCC_COLL_TYPE_BARRIER;

  CHECK_UCC_OK(ucc_collective_init(&args_, &req, team))

  CHECK_UCC_OK(ucc_collective_post(req));

  ucc_status_t status;
  while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
    ucc_context_progress(ctx);
  }
  CHECK_UCC_OK(status)

  return ucc_collective_finalize(req);
}

}