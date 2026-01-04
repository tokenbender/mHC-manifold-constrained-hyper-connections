# Handover: NS Coeff Schedule (NanoGPT-medium)

**Date:** 2026-01-04  
**Branch:** main

---

## Request

Switch Newton-Schulz coefficients from fixed quintic to the NanoGPT-medium schedule (list of per-step coefficients).

---

## Current State

- Default NS coeffs are quintic `(3.0, -3.2, 1.2)`.
- `zeropower_via_newtonschulz` and `orthostochastic_project` accept a single `(a, b, c)` tuple.
- `HyperConnections.__init__` wires `ns_coeffs` and passes through config.
- `test_mhc_orthostochastic_constraints` uses `hc.ns_coeffs` and checks orthogonality.

### Relevant Files

- `hyper_connections/hyper_connections.py`
  - `zeropower_via_newtonschulz(..., coeffs=(3.0, -3.2, 1.2))`
  - `orthostochastic_project(..., ns_coeffs=(3.0, -3.2, 1.2))`
- `examples/nanogpt/model.py`
  - `hc_kwargs` passes `ns_coeffs` from config
- `examples/nanogpt/train.py`
  - default `ns_coeffs = (3.0, -3.2, 1.2)`
- `examples/nanogpt/config/train_fineweb10B_mhc.py`
  - `ns_coeffs = (3.0, -3.2, 1.2)`
- `tests/test_hyper_connections.py`
  - `test_mhc_orthostochastic_constraints` uses `hc.ns_coeffs`

---

## Target Coeff Schedule (from NanoGPT-medium)

```
NS_COEFFS = [
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
]
```

---

## Implementation Notes

- Likely update `zeropower_via_newtonschulz` to accept either:
  - a single `(a, b, c)` tuple, or
  - a list of `(a, b, c)` tuples used per iteration.
- Decide behavior when schedule is provided:
  - ignore `steps`, or
  - require `steps == len(schedule)`.
- Ensure `orthostochastic_project` and `HyperConnections` propagate schedule.
- Update nanoGPT config to set the schedule as default for mHC runs.

---

## Open Questions

1. Should the schedule become the default `ns_coeffs`, or only set in configs?
2. If schedule length differs from `ns_steps`, should we error or slice?
3. Should tests enforce orthogonality with tighter tolerance when schedule is used?

---

## Validation

- `pytest tests/`
