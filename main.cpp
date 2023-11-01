#include <iostream>
#include <mpi.h>
#include <ucc/api/ucc.h>

#define CHECK_UCC_OK(expr) \
    if (const auto& r = (expr); r != UCC_OK) {                                              \
    std::cerr << "UCC error: " << r << " in " << __FILE__ << ":" << __LINE__ << std::endl;  \
    return r;                                                                               \
}

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

char ctx_type[] = "CTX ";
char team_type[] = "TEAM ";

template<const char *Type>
ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen, void *coll_info, void **req) {
    auto comm = static_cast<MPI_Comm>(coll_info);
    std::cout << Type << get_mpi_rank(comm) << " oob allgather " << msglen << std::endl;
    MPI_Request request;

    if (MPI_Iallgather(sbuf, (int) msglen, MPI_BYTE, rbuf, (int) msglen, MPI_BYTE, comm,
                       &request) != MPI_SUCCESS) {
        return UCC_ERR_NO_MESSAGE;
    }
    *req = request;
    return UCC_OK;
}

ucc_status_t oob_allgather_test(void *req) {
    auto request = static_cast<MPI_Request>(req);
    int completed;

    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
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

int main(int argc, char *argv[]) {
    // Initialise MPI and check its completion
    MPI_Init(&argc, &argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Get the rank for checking send to self, and initializations
    int32_t rank = get_mpi_rank(mpi_comm), world_size = get_mpi_world_size(mpi_comm);
    std::cout << "rank " << rank << " sz " << world_size << std::endl;

    ucc_lib_h lib;
    CHECK_UCC_OK(init_ucc(&lib))

    ucc_context_h ucc_ctx;
    CHECK_UCC_OK(create_ucc_ctx(lib, rank, world_size, &ucc_ctx))

    ucc_team_h ucc_team;
    CHECK_UCC_OK(create_ucc_team(ucc_ctx, rank, world_size, &ucc_team))

    MPI_Finalize();
    return 0;
}
