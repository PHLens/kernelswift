#include <cuda_runtime.h>

extern "C" __device__ __attribute__((always_inline, used))
unsigned long long corex_clock64_start() {
  return clock64();
}

extern "C" __device__ __attribute__((always_inline, used))
unsigned long long corex_clock64_after_u64(unsigned long long token) {
  if (token & 1ULL) {
    return clock64() + 1ULL;
  }
  return clock64();
}
