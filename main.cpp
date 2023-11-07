#include <iostream>
#include <vector>

#include "bench.hpp"

int main(int argc, char *argv[]) {
    // Initialise MPI and check its completion
    MPI_Init(&argc, &argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    std::string name(argv[1]);

    std::cout << "bench: " << name << std::endl;
    std::vector<std::string> bench_args;
    for (int i = 2; i < argc; i++) {
        bench_args.emplace_back(argv[i]);
    }

    auto bench = create_bench(name, mpi_comm, &bench_args);
    if (bench) {
        CHECK_UCC_OK(bench->Run());
    } else {
        throw std::runtime_error("Cannot find bench " + name);
    }

    MPI_Finalize();
    return 0;
}
