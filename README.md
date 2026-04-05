# Part 3: Discussion Questions

## 1. Pointers vs. References in C++

Pointers and references both provide indirect access to data, but they are not the same. A pointer is an object that stores a memory address. It can be reassigned, it can be `nullptr`, and it supports pointer arithmetic. A reference is an alias for an existing object. It must be initialized when it is created, it cannot later refer to a different object, and it is normally assumed to be valid.

In numerical algorithms, pointers are often more practical for low-level array and matrix kernels. In our implementation, all four multiply functions use `const double*` and `double*` because the data is stored in contiguous memory and accessed through manual indexing. This matches the style of BLAS-like routines and makes it easy to pass raw buffers, allocate dynamically, and test aligned versus unaligned memory.

References are usually a better choice when a function must receive a valid single object and should not take ownership or be reseated. For example, if we wrapped a matrix in a `Matrix` class, then helper functions such as `void printMatrix(const Matrix& A)` or `double& at(int i, int j)` would be natural uses of references. In short, we would choose pointers for raw numeric buffers and optional/null-checked data, and references for higher-level interfaces where aliasing a guaranteed-valid object is clearer.

## 2. Row-Major vs. Column-Major Storage and Cache Locality

Row-major order stores each row contiguously in memory, while column-major order stores each column contiguously. This directly affects whether the CPU reads data in a sequential way or with large memory strides.

In `multiply_mv_row_major`, the inner loop is

```cpp
result[i] += matrix[i * cols + j] * vector[j];
```

For a fixed row `i`, `j` moves left to right through contiguous memory. That is cache-friendly because nearby elements are loaded into the same cache line. In `multiply_mv_col_major`, the inner loop is

```cpp
result[i] += matrix[j * rows + i] * vector[j];
```

Now, as `j` changes, the matrix access jumps by `rows`. That creates a strided access pattern, which is less friendly to the cache, especially for large matrices. For that reason, row-major matrix-vector multiplication is typically faster when the matrix is stored in row-major order and the loop order matches the layout.

The same issue appears even more strongly in matrix-matrix multiplication. In `multiply_mm_naive`, the access to `matrixA[i * colsA + k]` is contiguous in `k`, but `matrixB[k * colsB + j]` walks down a column of `B`. In row-major storage, that means large jumps in memory. Our optimized version `multiply_mm_transposed_b` changes this by storing `B^T` and then using

```cpp
matrixB_transposed[j * rowsB + k]
```

which is contiguous in `k`. This makes both operands in the inner loop much more cache-friendly. The benchmark structure in `src.cpp` was designed to compare exactly this effect across sizes `32`, `128`, `512`, and `1024`, and our measured results clearly showed that the advantage of the transposed-`B` version grows with matrix size.

Using the `-O3` build, the measured unaligned timings were:

| Size | mv_row (us) | mv_col (us) | mm_naive (us) | mm_transposed (us) |
| --- | ---: | ---: | ---: | ---: |
| 32x32 | 0.225 | 0.215 | 7.75 | 7.72 |
| 128x128 | 3.79 | 10.795 | 1128.33 | 653.56 |
| 512x512 | 89.555 | 394.87 | 186575 | 49261.1 |
| 1024x1024 | 382.11 | 2835.22 | 3.22273e+06 | 404151 |

These numbers show two important effects. First, row-major matrix-vector multiplication becomes much faster than the column-major access pattern once the matrix is large enough to stress the cache. Second, transposing `B` before matrix-matrix multiplication has a dramatic impact at large sizes. At `1024x1024`, the transposed version was about eight times faster than the naive version, which is exactly what we would expect from better cache locality in the inner loop.

## 3. CPU Caches and Locality

Modern CPUs use a hierarchy of caches:

- L1 cache is the smallest and fastest, usually private to each core.
- L2 cache is larger and a little slower, still usually private to a core.
- L3 cache is the largest and slowest cache, often shared by several cores.

The goal is to keep frequently used data close to the processor so that the CPU does not have to wait for main memory, which is much slower.

Two key ideas are temporal locality and spatial locality. Temporal locality means that if data was used recently, it is likely to be used again soon. Spatial locality means that if one memory location is used, nearby locations are likely to be used soon as well.

We tried to exploit spatial locality mainly through access order. In `multiply_mv_row_major`, the matrix row is accessed sequentially. In `multiply_mm_transposed_b`, both the row of `A` and the corresponding row of `B^T` are read sequentially in the inner loop. We also exploit temporal locality because each `result[i]` or `result[i * colsB + j]` is updated repeatedly inside the inner loop before moving on. Keeping the accumulation local reduces unnecessary traffic to memory.

## 4. Memory Alignment

Memory alignment means placing data at addresses that are multiples of some boundary, such as 16, 32, or 64 bytes. Alignment matters because CPUs fetch memory in cache-line-sized chunks and many SIMD/vector instructions work best, or are simplest for the compiler to generate, when data begins at aligned addresses.

Our benchmark code includes a second experiment using 64-byte aligned memory to compare aligned and unaligned allocations. In principle, aligned data can reduce split loads and help the compiler generate more efficient vectorized code. However, our measurements showed that alignment alone produced only modest and inconsistent gains compared with changing the access pattern.

For example, in the `-O3` build, `mm_naive` improved from `186575 us` to `181850 us` at `512x512`, and from `3.22273e+06 us` to `3.01258e+06 us` at `1024x1024`. The transposed version changed only slightly, from `49261.1 us` to `49160.3 us` at `512x512`, and from `404151 us` to `403355 us` at `1024x1024`. In matrix-vector multiplication, alignment sometimes helped and sometimes slightly hurt. That means the dominant performance factor was still memory access order, not alignment by itself.

That matches the overall conclusion of this project: the biggest gains came from changing access patterns, especially transposing `B` before matrix-matrix multiplication. Alignment was useful as a supporting optimization, but not the main reason for speedup. If the compiler cannot vectorize effectively or if cache misses dominate runtime, alignment by itself will not solve the larger performance problem.

## 5. Compiler Optimizations

Compiler optimizations are extremely important in numerical code. At higher optimization levels, the compiler can inline small functions, remove redundant loads and stores, unroll loops, perform constant propagation, and sometimes auto-vectorize loops.

For our code, the baseline kernels are simple enough that optimization level has a major impact. At `-O0`, loop-heavy code often suffers from extra instructions, less aggressive register use, and no vectorization. At `-O2` or `-O3`, the same loops can become much faster. The optimized implementation with transposed `B` benefits twice: first from the better memory access pattern, and second from the compiler being able to optimize a cleaner inner loop.

We measured both `-O0` and `-O3`, and the difference was very large. Some representative unaligned results are:

| Size | Kernel | `-O0` (us) | `-O3` (us) |
| --- | --- | ---: | ---: |
| 128x128 | `mv_row` | 36.505 | 3.79 |
| 512x512 | `mv_col` | 1128.08 | 394.87 |
| 512x512 | `mm_naive` | 490491 | 186575 |
| 512x512 | `mm_transposed` | 311880 | 49261.1 |
| 1024x1024 | `mm_naive` | 3.18416e+06 | 3.22273e+06 |
| 1024x1024 | `mm_transposed` | 2.50223e+06 | 404151 |

The most striking result is that `-O3` helped the cache-friendly kernels much more than the cache-unfriendly ones. For `1024x1024`, `mm_transposed` dropped from about `2.50 s` to about `0.40 s`, while `mm_naive` stayed around `3.2 s`. This suggests that once a kernel is dominated by poor memory access, compiler optimization alone cannot fully rescue it.

Aggressive optimization also has drawbacks. It can increase compile time, make debugging harder, and sometimes produce larger binaries. In rare cases, it can also expose undefined behavior that happened to "work" in debug builds. So high optimization is valuable, but it should be combined with correctness testing and profiling rather than used blindly.

## 6. Profiling and Bottlenecks

The main bottlenecks in the initial implementations were memory-access related rather than arithmetic related. In particular:

- `multiply_mv_col_major` suffers from strided matrix access.
- `multiply_mm_naive` reads `B` down columns in row-major storage.
- Large problem sizes eventually exceed the small caches, so cache misses become much more expensive.

Profiling and benchmarking guided the optimization effort toward data layout and loop access order instead of micro-optimizing arithmetic. The most important lesson was that the cost of multiplication itself was not the limiting factor; the real issue was how efficiently the program moved data through the memory hierarchy. That is why pre-transposing `B` was a stronger optimization than small syntax-level changes.

Our measured results support that conclusion. At `512x512` with `-O3`, `mm_naive` took `186575 us`, while `mm_transposed` took only `49261.1 us`. At `1024x1024`, the gap became even larger: `3.22273e+06 us` versus `404151 us`. Those numbers show that the bottleneck in the naive version was not simply the number of floating-point operations; it was the cost of repeatedly accessing `B` with poor locality.

## 7. Teamwork Reflection

Dividing the initial implementation work was a good way to make fast progress. Different team members could focus on separate kernels, such as matrix-vector multiplication, matrix-matrix multiplication, benchmarking, and correctness checks. After that, collaborating on analysis and optimization helped us compare ideas and see the larger performance picture.

One challenge of this approach is consistency. Different people may use different naming styles, assumptions about storage order, or different testing strategies. Another challenge is that performance work is interconnected: one person may implement a correct function, but another person may notice that its loop order is inefficient. That means the second phase requires more discussion and shared understanding than the first.

The benefit is that teamwork gives both breadth and depth. We could parallelize the initial coding work, then combine perspectives during profiling and optimization. This made it easier to catch mistakes, interpret benchmark trends, and connect low-level implementation details to larger concepts such as cache locality, alignment, and compiler behavior.

## Summary

The main takeaway from this assignment is that high performance in numerical computing depends heavily on memory behavior. Pointer-based contiguous storage made the kernels simple to implement, but the true speed differences came from storage order, cache locality, and compiler optimization. Among our implementations, the most important optimization was changing matrix-matrix multiplication so that the inner loop accesses both inputs sequentially by using a transposed copy of `B`.
