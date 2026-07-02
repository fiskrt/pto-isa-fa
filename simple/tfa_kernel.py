"""Thin ctypes wrapper around libtfa_torch.so (the FA kernel's C-ABI launcher).

All FFI details live here so callers work in terms of plain ints and raw device
pointers. S0 (rows) and S1 (cols) are runtime arguments; only the head size and the
tiling granularity are fixed at build time (see tfa_config()).
"""
import ctypes
import os

DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "lib", "libtfa_torch.so")


class TfaKernel:
    """Loads libtfa_torch.so and exposes tfa_config / workspace_size / run."""

    def __init__(self, lib_path=DEFAULT_LIB):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"missing {lib_path}; build first: bash build.sh")
        self.lib_path = lib_path
        self._lib = ctypes.CDLL(lib_path)
        self._bind()

    def _bind(self):
        lib = self._lib
        lib.tfa_config.restype = None
        lib.tfa_config.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        lib.tfa_workspace_size.restype = ctypes.c_size_t
        lib.tfa_workspace_size.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.tfa_run.restype = ctypes.c_int
        lib.tfa_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.c_int]

    @property
    def config(self):
        """(head, s0_multiple, s1_multiple): fixed head size and the required multiples for S0/S1."""
        head, s0m, s1m = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        self._lib.tfa_config(ctypes.byref(head), ctypes.byref(s0m), ctypes.byref(s1m))
        return head.value, s0m.value, s1m.value

    def validate_shape(self, s0, s1):
        """Raise ValueError unless (s0, s1) satisfy the kernel's tiling multiples."""
        _, s0m, s1m = self.config
        if s0 <= 0 or s0 % s0m:
            raise ValueError(f"S0={s0} must be a positive multiple of {s0m}")
        if s1 <= 0 or s1 % s1m:
            raise ValueError(f"S1={s1} must be a positive multiple of {s1m}")

    def workspace_size(self, s0, s1):
        """Bytes to allocate for the workspace block for this shape."""
        return int(self._lib.tfa_workspace_size(s0, s1))

    def run(self, q_ptr, k_ptr, v_ptr, o_ptr, ws_ptr, stream, s0, s1, causal=False):
        """Enqueue the kernel on `stream`. Pointers are raw device addresses (ints from data_ptr()).

        `causal` selects the lower-triangular masked variant. Returns the launcher's rc
        (0 on enqueue when a caller stream is given).
        """
        return self._lib.tfa_run(q_ptr, k_ptr, v_ptr, o_ptr, ws_ptr, stream, s0, s1, int(causal))
