"""Native zkgemm extension import helper."""

from __future__ import annotations


def import_zkgemm_cuda():
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    import zkgemm_cuda

    return zkgemm_cuda
