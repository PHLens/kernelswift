/* Copyright (c) 2026 kernelswift authors.
 *
 * Host driver for matmul_probe — verifies __bang_matmul fp32 overload.
 * Tests two src1 layouts: [K, N] row-major (matmul natural) vs [N, K]
 * row-major (conv kernel layout, since the underlying intrinsic is a 1x1 conv).
 */

#include <cnrt.h>
#include "/projs/framework/lipenghui/neuware_home/lib/clang/11.1.0/include/bang_fp16.h"
#include "/projs/framework/lipenghui/neuware_home/lib/clang/11.1.0/include/bang_host_functions_decls.h"
typedef __half half;

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define M (64)
#define K (64)
#define N (64)

extern "C" void launch_probe(float *src0, float *src1, float *dst,
                             int trans_en, cnrtQueue_t queue);

static void cnrt_check(cnrtRet_t ret, const char *where) {
  if (ret != cnrtSuccess) {
    fprintf(stderr, "[cnrt error] %s: ret=%d\n", where, (int)ret);
    exit(1);
  }
}

// dst[M, N] = src0[M, K] @ src1_logical[K, N]
// where src1_buf holds src1_logical in some physical layout.
static void run_and_report(const char *label, const std::vector<float> &src0,
                           const std::vector<float> &src1_buf, int trans_en,
                           const std::vector<float> &expected,
                           cnrtQueue_t queue, float *d_src0, float *d_src1,
                           float *d_out) {
  cnrt_check(cnrtMemcpy(d_src0, const_cast<float *>(src0.data()),
                        M * K * sizeof(float), cnrtMemcpyHostToDev),
             "memcpy src0");
  cnrt_check(cnrtMemcpy(d_src1, const_cast<float *>(src1_buf.data()),
                        K * N * sizeof(float), cnrtMemcpyHostToDev),
             "memcpy src1");

  launch_probe(d_src0, d_src1, d_out, trans_en, queue);
  cnrt_check(cnrtQueueSync(queue), "sync");

  std::vector<float> out(M * N, 0.f);
  cnrt_check(cnrtMemcpy(out.data(), d_out, M * N * sizeof(float),
                        cnrtMemcpyDevToHost), "memcpy out");

  float max_diff = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float d = fabsf(out[i] - expected[i]);
    if (d > max_diff) max_diff = d;
  }
  printf("== %s (trans_en=%d) ==\n", label, trans_en);
  printf("expected: ");
  for (int i = 0; i < M * N; ++i) printf("%.2f ", expected[i]);
  printf("\ngot:      ");
  for (int i = 0; i < M * N; ++i) printf("%.2f ", out[i]);
  printf("\nmax_abs_diff: %.4f -> %s\n\n", max_diff,
         max_diff < 1e-3f ? "PASS" : "FAIL");
}

int main() {
  // src0 = [1..M*K]  (M=64, K=64). Row m has values [m*64+1 .. m*64+64].
  std::vector<float> h_src0(M * K);
  for (int i = 0; i < M * K; ++i) h_src0[i] = (float)(i + 1);

  // Test 1: all-ones src1, [K, N] row-major.
  // dst[m, n] = sum_k src0[m, k] * 1 = sum(row m of src0) = sum(m*64+1..m*64+64) = 64*m*64 + 2080
  std::vector<float> h_src1_KN(K * N, 1.f);
  std::vector<float> expected_all_ones(M * N);
  for (int m = 0; m < M; ++m) {
    float row_sum = 0.f;
    for (int k = 0; k < K; ++k) row_sum += h_src0[m * K + k];
    for (int n = 0; n < N; ++n) expected_all_ones[m * N + n] = row_sum;
  }

  // Test 2: all-ones src1, [N, K] row-major (conv kernel layout).
  // Same expected (all-ones is layout-symmetric).
  std::vector<float> h_src1_NK(K * N, 1.f);  // values identical, but semantically [N,K]

  // Test 3: identity src1 as [K, N] (src1[k,n] = 1 if k==n).
  // dst[m, n] = src0[m, n] -> row m = [m*64+1 .. m*64+64]
  std::vector<float> h_src1_ident_KN(K * N, 0.f);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      h_src1_ident_KN[k * N + n] = (k == n) ? 1.f : 0.f;
  std::vector<float> expected_ident(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      expected_ident[m * N + n] = h_src0[m * K + n];

  // Test 4: identity src1 as [N, K] (src1[n,k] = 1 if k==n).
  // Same expected.
  std::vector<float> h_src1_ident_NK(K * N, 0.f);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      h_src1_ident_NK[n * K + k] = (k == n) ? 1.f : 0.f;

  float *d_src0, *d_src1, *d_out;
  cnrt_check(cnrtMalloc((void **)&d_src0, M * K * sizeof(float)), "malloc src0");
  cnrt_check(cnrtMalloc((void **)&d_src1, K * N * sizeof(float)), "malloc src1");
  cnrt_check(cnrtMalloc((void **)&d_out, M * N * sizeof(float)), "malloc out");

  cnrtQueue_t queue;
  cnrt_check(cnrtQueueCreate(&queue), "queue create");

  printf("=== __bang_matmul fp32 probe (M=%d K=%d N=%d) ===\n\n", M, K, N);
  // Test both trans_en values for the [K,N] layout (natural matmul layout).
  run_and_report("all-ones src1, [K,N] row-major", h_src0, h_src1_KN, 0,
                 expected_all_ones, queue, d_src0, d_src1, d_out);
  run_and_report("all-ones src1, [K,N] row-major", h_src0, h_src1_KN, 1,
                 expected_all_ones, queue, d_src0, d_src1, d_out);
  run_and_report("identity src1, [K,N] row-major", h_src0, h_src1_ident_KN, 0,
                 expected_ident, queue, d_src0, d_src1, d_out);
  run_and_report("identity src1, [K,N] row-major", h_src0, h_src1_ident_KN, 1,
                 expected_ident, queue, d_src0, d_src1, d_out);

  cnrtFree(d_src0);
  cnrtFree(d_src1);
  cnrtFree(d_out);
  cnrtQueueDestroy(queue);
  return 0;
}
