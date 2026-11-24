# Stale: attribution patching with a truncated gradient

These two files were made by the first version of `experiments/atp_baseline.py`, which detached
every layer's MLP output and so dropped every path through later MLPs (DECISIONS.md D10,
deviation of 2026-11-24; `tests/test_atp.py`). They are kept only so the old report numbers
can be traced. No analysis reads them. The corrected files are recomputed on the server and
written to `experiments/results/` under the same names.
