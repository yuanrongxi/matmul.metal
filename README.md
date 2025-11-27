# matmul.metal
matmul.metal is a matrix multiplication implementation using Metal on macOS. It includes implementations of matrix multiplication via CPU, GPU, and Metal MPS, and calculates the GFlops for each algorithm. Its primary purpose is to study Metal and GPU programming methods

## Setup
The project compilation depends on the OpenMP library, which needs to be installed before building. The installation methods are as follows:
```bash
brew install libomp
```
## Compile and Run
```bash
git clone git clone https://github.com/yuanrongxi/matmul.metal.git
cd ./matmul.metal
cmake .
make
./matmul
```

## Overview
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to MPS    |
|:------------------------------------|----------:|:-------------------------------|
| 1: CPU Naive                        |  `132.8` | 4.3%                           |
| 2: CPU block tiling                 |  `168.6` | 5.5%                           |
| 3: Metal Native                     |  `347.1` | 11.2%                          |
| 4: Metal Block cache                |  `723.8` | 23.4%                          |
| 5: Metal 1D Blocktiling             | `1126.7` | 36.4%                          |
| 7: Metal 2D Blocktiling             | `2168.9` | 70.2%                          |
| 6: Metal Vectorized Mem Access      | `2334.2` | 75.5%                          |
| 8: Warptiling                       | `2129.3` | 68.9%                          |
| 9: python numpy(GPU)                | `1295.3` | 41.9%                          |
| 0: Metal Performance Shaders(MPS)   | `3091.5` | 100.0%                         |
<!-- benchmark_results -->
