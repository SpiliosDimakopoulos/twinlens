"""TwinLens: a digital twin of GPT-2's internals that measures how far its explanations hold."""

import os

# numpy is only used for memory-mapping; its OpenBLAS thread pool would otherwise reserve
# per-thread buffers that matter on low-memory machines. torch keeps its own thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
