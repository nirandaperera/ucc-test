//
// Created by niranda on 11/1/23.
//
#pragma once

#include <memory>
#include <chrono>
#include "utils.hpp"

class Benchmark {
public:
    Benchmark(MPI_Comm comm, int iter) : rank(get_mpi_rank(comm)), world_size(get_mpi_world_size(comm)),
                                         iter(iter) {
        std::cout << "rank " << rank << " sz " << world_size << std::endl;
    }

    virtual ~Benchmark() = default;

    virtual ucc_status_t Run() = 0;

protected:
    int rank, world_size, iter;
};

class CtxCreateBenchmark : public Benchmark {
public:
    CtxCreateBenchmark(MPI_Comm comm, int iter) : Benchmark(comm, iter) {}

    static std::string_view Name() { return "ctx_create"; }

    ucc_status_t DoRun(std::array<double, 4> &timings) {
        ucc_lib_h lib;
        CHECK_UCC_OK(init_ucc(&lib))

        ucc_context_h ucc_ctx;
        CHECK_UCC_OK(create_ucc_ctx(lib, rank, world_size, &ucc_ctx))

        ucc_team_h ucc_team;
        CHECK_UCC_OK(create_ucc_team(ucc_ctx, rank, world_size, &ucc_team))

        return UCC_OK;
    }

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


std::unique_ptr<Benchmark> create_bench(const std::string_view &name, MPI_Comm comm, int iter) {
    if (name == CtxCreateBenchmark::Name()) {
        return std::make_unique<CtxCreateBenchmark>(comm, iter);
    }

    return nullptr;
}