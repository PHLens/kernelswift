# Round 008 Verification Status

- phase: measurement-exclusive named probe
- round: `008`
- status: probe complete; no candidate verification or terminal report
- accepted reference: `reference_triton_grouped_topk_003.py`
- canonical candidate: `triton_grouped_topk_003.py`
- reference SHA-256: `9977aaf9ec96c851be33f2582e6284451fd41686a1acc4607deb4e104dca5ea7`
- canonical SHA-256: `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37`
- measurement fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`

## Completed Commands

Remote benchmark was already completed successfully and was not rerun during recovery:

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_003.py --warmup 50 --repeat 100 --full-traceback
```

Recorded raw timing pair:

```text
PASS accuracy; v0=0.282114 ms, v1=0.282032 ms, speedup=1.000x
Summary: 1 passed, 0 failed, 1 total.
```

The matching remote forward profile was also already completed successfully and was not rerun:

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_003.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_probe_round008_forward_50iter.pt.trace.json
```

Trace copy command used for recovery:

```bash
/tmp/kernelswift-sshpass/extracted/usr/bin/sshpass -f /dev/shm/kernelswift-s60-sshpass scp -P 32222 root+vm-ngFfDbX7jgamQdwJ@106.13.250.54:/root/kernelswift-s60/log/groupedtopk_probe_round008_forward_50iter.pt.trace.json log/groupedtopk_probe_round008_forward_50iter.pt.trace.json
```

Local summary command:

```bash
python3 /home/phlens/kernelswift/.dsh/skills/kernel-opt-loop/scripts/summarize_trace.py log/groupedtopk_probe_round008_forward_50iter.pt.trace.json --iterations 50
```

## Trace Evidence

- trace: `log/groupedtopk_probe_round008_forward_50iter.pt.trace.json`
- trace bytes: `194948`
- trace SHA-256: `1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec`
- iterations: `50`
- device_time_available: `false`
- device_time_reason: `GCU trace exposes runtime launch events but no cat=kernel device durations`
- device total us: unavailable
- device us/call: unavailable
- kernel count: unavailable
- runtime launch count: `100` total, `2.0` per call
- runtime launch total: `1171.80908203125 us`
- runtime launch us/call: `23.436181640625 us`
- runtime launch: `topsModuleLaunchKernel`

Runtime launch duration is diagnostic only and is not device kernel duration. No device ratio or kernel-time claim is made.

## Probe Result And Next Safe Action

The named probe completed: correctness passed and the recorded single benchmark pair was reference `0.282114 ms` versus canonical `0.282032 ms`. This measurement-only evidence does not produce an optimization result or terminal classification. Preserve the accepted canonical pointer and route the trace evidence to the next Designer decision through Orchestrator after the measurement-exclusive boundary is durably closed.
