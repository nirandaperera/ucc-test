# UCC Test 

## Instructions 

1. Create conda env 
    ```bash
    conda create -f conda/ucc.yml
    conda activate ucc
    ````
2. Build ucc 
    ```bash
    git clone https://github.com/openucx/ucc.git
    cd ucc
    ./autogen.sh
    ./configure --prefix=$PWD/install --with-ucx=$CONDA_PREFIX
    make -j install 
    ./install/bin/ucc_info -vbaA
    ```
3. Build project 
   ```bash
   git clone https://github.com/nirandaperera/ucc-test.git
   mkdir build 
   cd build
   cmake -DMPI_CXX_COMPILER=<mpicxx path> \
       -DUCX_INSTALL_PREFIX=$CONDA_PREFIX/lib/cmake/ucx \
       -DUCC_INSTALL_PREFIX=<ucc path>/install \
       ..
   make -j
   ```
4. Run a test 
   ```bash
   mpirun -x UCX_TLS=<tp> -x UCX_NET_DEVICES=<nic> -mca btl_tcp_if_include <nic> -mca btl tcp,self -n 8 ucc_test <args ...>
   ```