#include "bench.hpp"

int main(int argc, char *argv[]) {
    // Initialise MPI and check its completion
    MPI_Init(&argc, &argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    std::string_view name(argv[1]);
    int iter = atoi(argv[2]);

    if (argc != 3) {
        std::cout << "ERROR: num args != 2" << std::endl;
        return -1;
    }

    std::cout << "bench: " << name << " iter: " << iter << std::endl;
    auto bench = create_bench(name, mpi_comm, iter);
    if (bench) {
        CHECK_UCC_OK(bench->Run());
    } else {
        std::cout << "ERROR: cannot find bench " << name << std::endl;
        return -1;
    }

    MPI_Finalize();
    return 0;
}
