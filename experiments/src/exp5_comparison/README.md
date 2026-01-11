# 🧪 Experiment 5: Comparative Identity Proof

This is the core diagnostic module for proving the structural benefits of **RLAE** over traditional weight mutation.

## Theoretical Sections

- **SEC1 (Unstructured Mutation):** Simulates training drift via random noise. Proves that "Identity Scars" are irreversible.
- **SEC2 (Structured Mutation):** Real gradient-based weight adaptation. Proves that optimizer-driven changes create permanent structural patterns.
- **SEC3 (RLAE Method):** The solution. Demonstrates that behavioral parameters can be mounted and unmounted, restoring the model to a 100% pure state.

## 🏁 Critical Validation Runs (M-Series)

### **M1 — Repeatability Run**

Locked to `seed=1337`.  
**Command:** `python irreversibility_test.py`  
**Goal:** Verify that the RF (Recoverability Factor) for SEC3 is statistically superior to SEC1/SEC2 in a deterministic baseline.

### **M2 — No-Op Control**

Metric grounding (Zero-Point verification).  
**Command:** `python irreversibility_test.py --control`  
**Goal:** Prove that the measurement pipeline itself introduces no divergence (`KL ≈ 0`).

---

## Technical Features

- **Deterministic Noise:** Mutations are generated using a fixed random state.
- **PEFT Integration:** SEC3 uses official PEFT `unload()` mechanisms to prove theoretical reversibility.
- **Multi-Metric Logging:** Captures KL Divergence and Recoverability Factor (RF) for every scenario.
