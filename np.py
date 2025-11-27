import numpy as np
import time
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

def np_matmul(m, n, k):
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    c = np.zeros((m, n), dtype=np.float32)
    start = time.perf_counter()
    for i in range(100):
        c = np.dot(a, b)
    end = time.perf_counter()
    cost = (end - start)/100
    gflops = 2.0 * m * n * k / cost / 1024 / 1024 / 1024
    print(f"GPU matmul Time taken: {cost:.4f} s, GFLOPS: {gflops:.2f}")


np_matmul(1024, 1024, 1024)
np_matmul(64, 64, 64)


