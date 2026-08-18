# Round Status 004

Status: verifying

- Candidate: `triton_grouped_topk_004.py`
- Candidate SHA-256: `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83`
- Decision: `rounds/decision_004.md`
- Decision SHA-256: `307f4a03c15b08daca8bb571f0391418997a07d864ff357b9f2d113cf2fb8f65`
- Reference: `baseline_adapter.py`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- Measurement exclusive: `true`
- Correctness: pass; unchanged harness and adversarial tie suite passed on BI150.
- Screening: pass; corrected base-vs-candidate pairs showed consistent candidate speedup.
- Timing: pass; canonical-adapter paired medians are baseline `0.466908 ms`, candidate `0.432098 ms`, improvement `7.455430192%`.
- Profiler: pass; BI150 trace exposes device durations. Reference `178.991259765625 us/call`, `14.86 kernels/call`; candidate `127.260771484375 us/call`, `9.9 kernels/call`.

Correctness, wall-time, and targeted profiler gates passed. Candidate is eligible for adoption evaluation.
