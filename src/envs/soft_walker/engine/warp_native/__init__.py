"""Native Warp spring, beam, FEM and 2D/3D MPM implementations.

Importing this namespace does not import JAX. Portable NumPy bundles are
instantiated at reset; resident CUDA graphs own whole forward/adjoint execution.
Unsupported capabilities raise explicitly; CUDA mode has no CPU/JAX fallback.
The optional oracle modules import JAX only for export and parity validation.
"""
