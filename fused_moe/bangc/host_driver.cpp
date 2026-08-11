/* Copyright (c) 2026 kernelswift authors.
 *
 * Host driver for fused_moe BangC kernel.
 *
 * Generates inputs matching fused_moe/base.py (T=83, H=128, E=8, top_k=2, I=64),
 * pre-transposes w1 and w2 on CPU, launches FusedMoEKernel, measures device time
 * via cnrtNotifier and wall time via std::chrono, and validates against a CPU
 * fp32 reference with atol=5e-2.
 *
 * Build: see ../CMakeLists.txt (bang_add_executable links cnrt).
 * Run:   ./fused_moe_bangc
 */

#include <cnrt.h>

// Including bang_fp16.h via absolute path (not via -I path) keeps the clang
// include dir out of the system <stdint.h>/<stddef.h> search path. The clang
// versions of those headers use `__has_feature` (clang-only) and would break
// g++ when cnrt.h pulls them in transitively.
#include "/projs/framework/lipenghui/neuware_home/lib/clang/11.1.0/include/bang_fp16.h"
#include "/projs/framework/lipenghui/neuware_home/lib/clang/11.1.0/include/bang_host_functions_decls.h"
// `half` is a BangC device-side keyword; on host (g++), alias to __half so the
// kernel signature resolves.
typedef __half half;

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define H (128)
#define I (64)
#define TWO_I (128)
#define E (8)
#define K (2)
#define T (83)

extern "C" void launch_fused_moe(half *hidden, float *router_logits,
                                 half *w1_T, half *w2_T, half *out, int t_count,
                                 cnrtQueue_t queue);

void policyFunc(cnrtDim3_t *dim, cnrtFunctionType_t *func_type);

// ---------- helpers ----------

static void cnrt_check(cnrtRet_t ret, const char *where) {
  if (ret != cnrtSuccess) {
    fprintf(stderr, "[cnrt error] %s: ret=%d\n", where, (int)ret);
    exit(1);
  }
}

// CPU reference for fused MoE. Computes the same op as base.py:
//   scores = softmax(router_logits)
//   topk_weights, topk_ids = topk(scores, K)
//   topk_weights /= sum(topk_weights)
//   out = sum_k topk_weights[k] * (silu(x @ w1[e_k].T)[:I] * (x @ w1[e_k].T)[I:]
//                                  @ w2[e_k].T)
// All in fp32; fp16 weights are upcast to fp32 on CPU.
static void cpu_fused_moe(const half *hidden, const float *router_logits,
                          const half *w1, const half *w2, float *out) {
  for (int t = 0; t < T; ++t) {
    // softmax
    float logits[E];
    float mx = router_logits[t * E + 0];
    for (int e = 1; e < E; ++e) {
      if (router_logits[t * E + e] > mx) mx = router_logits[t * E + e];
    }
    float sc[E];
    float s = 0.f;
    for (int e = 0; e < E; ++e) {
      sc[e] = expf(router_logits[t * E + e] - mx);
      s += sc[e];
    }
    for (int e = 0; e < E; ++e) sc[e] /= s;

    // top-2 selection sort
    int ids[K];
    float vals[K];
    {
      int b0 = 0;
      float v0 = sc[0];
      for (int e = 1; e < E; ++e) {
        if (sc[e] > v0) { v0 = sc[e]; b0 = e; }
      }
      ids[0] = b0;
      vals[0] = v0;
      sc[b0] = -1e30f;
    }
    {
      int b1 = 0;
      float v1 = sc[0];
      for (int e = 1; e < E; ++e) {
        if (sc[e] > v1) { v1 = sc[e]; b1 = e; }
      }
      ids[1] = b1;
      vals[1] = v1;
    }

    // renorm
    float ws = vals[0] + vals[1];
    float w[K] = {vals[0] / ws, vals[1] / ws};

    // token hidden [H]
    float x[H];
    for (int h = 0; h < H; ++h) x[h] = (float)hidden[t * H + h];

    // accumulate output
    float out_acc[H] = {0.f};
    for (int k = 0; k < K; ++k) {
      int e = ids[k];
      // w1[e] is [2I, H] half; w1[e].T is [H, 2I]
      // gate_up = x @ w1[e].T  -> [2I]
      float gate_up[TWO_I] = {0.f};
      for (int n = 0; n < TWO_I; ++n) {
        float acc = 0.f;
        for (int kk = 0; kk < H; ++kk) {
          // w1[e, n, kk] -> index e*(2I*H) + n*H + kk
          acc += x[kk] * (float)w1[e * (TWO_I * H) + n * H + kk];
        }
        gate_up[n] = acc;
      }
      // act = silu(gate_up[:I]) * gate_up[I:]
      float act[I];
      for (int i = 0; i < I; ++i) {
        float g = gate_up[i];
        float u = gate_up[i + I];
        float sig = 1.f / (1.f + expf(-g));
        act[i] = g * sig * u;
      }
      // w2[e] is [H, I] half; w2[e].T is [I, H]
      // out_k = act @ w2[e].T  -> [H]
      float out_k[H];
      for (int h = 0; h < H; ++h) {
        float acc = 0.f;
        for (int kk = 0; kk < I; ++kk) {
          // w2[e, h, kk] -> e*(H*I) + h*I + kk
          acc += act[kk] * (float)w2[e * (H * I) + h * I + kk];
        }
        out_k[h] = acc * w[k];
      }
      for (int h = 0; h < H; ++h) out_acc[h] += out_k[h];
    }

    for (int h = 0; h < H; ++h) out[t * H + h] = out_acc[h];
  }
}

int main() {
  // ---------------- init ----------------
  std::mt19937 rng(20260810);
  // fp16 helper via fp32 round-trip. After P2 (intermediates promoted to
  // fp32), GEMM accumulation no longer underflows at std=0.02 — the only
  // fp16 in the device path is x/w (loaded as half from GDRAM) and the
  // final store. Using std=0.02 matches auto_bench.py's convention for
  // v5 Triton, making the atol=5e-2 criterion apples-to-apples. (Earlier
  // std=1.0 was a workaround for the pre-P2 fp16-intermediate bug; no
  // longer needed.)
  auto randn_f16 = [&](half &out) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    out = (half)nd(rng);
  };

  // host-side buffers
  std::vector<half> h_hidden(T * H);
  std::vector<float> h_router(T * E);
  std::vector<half> h_w1(E * TWO_I * H);   // [E, 2I, H]
  std::vector<half> h_w2(E * H * I);       // [E, H, I]
  std::vector<half> h_w1_T(E * H * TWO_I); // [E, H, 2I] (pre-transposed)
  std::vector<half> h_w2_T(E * I * H);     // [E, I, H]  (pre-transposed)
  std::vector<half> h_out(T * H, (half)0);
  std::vector<float> h_out_ref(T * H, 0.f);

  for (int i = 0; i < T * H; ++i) randn_f16(h_hidden[i]);
  for (int i = 0; i < T * E; ++i)
    h_router[i] = std::normal_distribution<float>(0.f, 1.f)(rng);
  for (int i = 0; i < E * TWO_I * H; ++i) randn_f16(h_w1[i]);
  for (int i = 0; i < E * H * I; ++i) randn_f16(h_w2[i]);

  // pre-transpose w1 [E, 2I, H] -> [E, H, 2I]  (in memory: dst[e, h, n] = src[e, n, h])
  for (int e = 0; e < E; ++e) {
    for (int n = 0; n < TWO_I; ++n) {
      for (int h = 0; h < H; ++h) {
        h_w1_T[e * (H * TWO_I) + h * TWO_I + n] =
            h_w1[e * (TWO_I * H) + n * H + h];
      }
    }
  }
  // pre-transpose w2 [E, H, I] -> [E, I, H]  (in memory: dst[e, i, h] = src[e, h, i])
  for (int e = 0; e < E; ++e) {
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < I; ++i) {
        h_w2_T[e * (I * H) + i * H + h] =
            h_w2[e * (H * I) + h * I + i];
      }
    }
  }

  // ---------------- device buffers ----------------
  half *d_hidden, *d_w1_T, *d_w2_T, *d_out;
  float *d_router;
  cnrt_check(cnrtMalloc((void **)&d_hidden, T * H * sizeof(half)), "cnrtMalloc hidden");
  cnrt_check(cnrtMalloc((void **)&d_router, T * E * sizeof(float)), "cnrtMalloc router");
  cnrt_check(cnrtMalloc((void **)&d_w1_T, E * H * TWO_I * sizeof(half)), "cnrtMalloc w1_T");
  cnrt_check(cnrtMalloc((void **)&d_w2_T, E * I * H * sizeof(half)), "cnrtMalloc w2_T");
  cnrt_check(cnrtMalloc((void **)&d_out, T * H * sizeof(half)), "cnrtMalloc out");

  cnrt_check(cnrtMemcpy(d_hidden, h_hidden.data(), T * H * sizeof(half),
                        cnrtMemcpyHostToDev), "cnrtMemcpy hidden");
  cnrt_check(cnrtMemcpy(d_router, h_router.data(), T * E * sizeof(float),
                        cnrtMemcpyHostToDev), "cnrtMemcpy router");
  cnrt_check(cnrtMemcpy(d_w1_T, h_w1_T.data(), E * H * TWO_I * sizeof(half),
                        cnrtMemcpyHostToDev), "cnrtMemcpy w1_T");
  cnrt_check(cnrtMemcpy(d_w2_T, h_w2_T.data(), E * I * H * sizeof(half),
                        cnrtMemcpyHostToDev), "cnrtMemcpy w2_T");

  // ---------------- queue + notifier ----------------
  cnrtQueue_t queue;
  cnrtNotifier_t start, end;
  cnrt_check(cnrtQueueCreate(&queue), "cnrtQueueCreate");
  cnrt_check(cnrtNotifierCreate(&start), "cnrtNotifierCreate");
  cnrt_check(cnrtNotifierCreate(&end), "cnrtNotifierCreate");

  cnrtDim3_t dim;
  cnrtFunctionType_t func_type;
  policyFunc(&dim, &func_type);
  int task_dim = dim.x * dim.y * dim.z;

  // ---------------- launch params ----------------
  // Launch is wrapped in .mlu (extern "C" launch_fused_moe) because the
  // <<<dim, func_type, queue>>> syntax is BangC-specific (cncc only).
  auto launch = [&]() {
    launch_fused_moe(d_hidden, d_router, d_w1_T, d_w2_T, d_out, T, queue);  };

  // ---------------- warmup ----------------
  for (int i = 0; i < 50; ++i) launch();
  cnrt_check(cnrtQueueSync(queue), "warmup sync");

  // ---------------- device-time measure ----------------
  const int repeat = 100;
  cnrt_check(cnrtQueueSync(queue), "pre-measure sync");
  cnrt_check(cnrtPlaceNotifier(start, queue), "place start");
  for (int i = 0; i < repeat; ++i) launch();
  cnrt_check(cnrtPlaceNotifier(end, queue), "place end");
  cnrt_check(cnrtQueueSync(queue), "measure sync");

  // cnrtNotifierDuration returns the elapsed time in MICROSECONDS (despite
  // the parameter name in the header). Divide by repeat to get per-iter us.
  float total_us = 0.f;
  cnrt_check(cnrtNotifierDuration(start, end, &total_us),
            "notifier duration");
  float device_us_per_iter = total_us / repeat;

  // ---------------- wall-time measure ----------------
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; ++i) launch();
  cnrt_check(cnrtQueueSync(queue), "wall sync");
  auto t1 = std::chrono::high_resolution_clock::now();
  double wall_us_per_iter =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / repeat;

  // ---------------- correctness ----------------
  cnrt_check(cnrtMemcpy(h_out.data(), d_out, T * H * sizeof(half),
                        cnrtMemcpyDevToHost), "cnrtMemcpy out");
  // debug: check inputs
  printf("h_hidden[0..3]: %.4e %.4e %.4e %.4e\n", (float)h_hidden[0],
         (float)h_hidden[1], (float)h_hidden[2], (float)h_hidden[3]);
  printf("h_router[0..3]: %.4e %.4e %.4e %.4e\n", h_router[0], h_router[1],
         h_router[2], h_router[3]);
  printf("h_w1[0..3]:     %.4e %.4e %.4e %.4e\n", (float)h_w1[0],
         (float)h_w1[1], (float)h_w1[2], (float)h_w1[3]);
  cpu_fused_moe(h_hidden.data(), h_router.data(), h_w1.data(),
                h_w2.data(), h_out_ref.data());

  float max_diff = 0.f;
  for (int i = 0; i < T * H; ++i) {
    float d = fabsf((float)h_out[i] - h_out_ref[i]);
    if (d > max_diff) max_diff = d;
  }
  // sanity prints: first 4 device outputs and first 4 cpu ref outputs
  printf("h_out[0..3]:  %.4e %.4e %.4e %.4e\n", (float)h_out[0], (float)h_out[1],
         (float)h_out[2], (float)h_out[3]);
  printf("h_ref[0..3]:  %.4e %.4e %.4e %.4e\n", h_out_ref[0], h_out_ref[1],
         h_out_ref[2], h_out_ref[3]);

  // ---------------- report ----------------
  printf("=== fused_moe BangC kernel ===\n");
  printf("shape: T=%d H=%d E=%d top_k=%d I=%d\n", T, H, E, K, I);
  printf("task_dim: %d\n", task_dim);
  printf("device time/iter: %.2f us  (avg of %d iters)\n",
         device_us_per_iter, repeat);
  printf("wall   time/iter: %.2f us  (avg of %d iters)\n",
         wall_us_per_iter, repeat);
  printf("max_abs_diff: %.4f  (atol=5e-2 -> %s)\n", max_diff,
         max_diff < 5e-2f ? "PASS" : "FAIL");

  // cleanup
  cnrtFree(d_hidden);
  cnrtFree(d_router);
  cnrtFree(d_w1_T);
  cnrtFree(d_w2_T);
  cnrtFree(d_out);
  cnrtNotifierDestroy(start);
  cnrtNotifierDestroy(end);
  cnrtQueueDestroy(queue);

  return 0;
}
