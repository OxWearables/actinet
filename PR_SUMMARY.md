# Retain test-set timestamps in cross-validation fold results

Cross-validation fold results now carry the timestamps of the test windows, letting downstream analysis align predictions to time.

## Files Changed
**`src/actinet/evaluate.py`**
- Add a `"time"` field to each per-fold result dictionary for both the ActiNet and random forest evaluations, reusing the already-computed per-fold test timestamps.
- When timestamps are unavailable, store a per-window sequence of `None`s matching the number of predictions instead of a scalar `None`, so every fold's fields stay uniform for downstream concatenation.
