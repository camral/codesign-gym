"""Convex decomposition of fingertip meshes, cached and run in worker processes.

MuJoCo collides every mesh geom as its convex hull, so a concave fingertip
is split into convex parts first. Each job is one fingertip mesh; identical
meshes are decomposed once, results are kept in an in-memory LRU cache keyed
by the mesh bytes, and identical parts are shared across worlds by naming
them after a digest of their content.

A backend aims for ``target_parts``. A mesh that still yields more than
``max_parts`` usable parts is retried down to 2; past that, VHACD fails the
job (``ConvexDecompositionError``), and CuACD/VisACD fall back to VHACD.

- CuACD (default on CUDA) runs a persistent multi-worker GPU pool with
  single-mesh VHACD fallback and pool recycling on kernel faults;
- VHACD runs one job per worker in a spawned CPU process pool;
- VisACD (optional CUDA/OptiX extension) runs all jobs as one native batch in
  a dedicated spawned worker.

- ConvexPart: one convex mesh, ready for ``MjSpec.add_mesh``;
- ConvexDecomposer: decompose a batch of jobs with caching;
- content_address: name parts by content and share duplicates.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import importlib
import multiprocessing
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing import shared_memory

import numpy as np
import trimesh

from ..config import MIN_PART_DIM_M


@dataclass(frozen=True)
class ConvexPart:
    """One convex mesh ready for direct insertion into an ``MjSpec``."""
    vertices: np.ndarray
    faces: np.ndarray
    name: str = ""


@dataclass(frozen=True)
class DecompositionJob:
    job_id: Hashable
    vertices: np.ndarray
    faces: np.ndarray
    mesh_name: str


class ConvexPartCapError(RuntimeError):
    """A mesh could not be decomposed within the hard part cap (or faulted in the backend)."""

    def __init__(self, backend: str, mesh_name: str, part_count: int, max_parts: int, target_parts: int,
                 detail: str | None = None):
        if detail:
            super().__init__(f"{backend.upper()} failed for {mesh_name}: {detail}")
        else:
            super().__init__(f"{backend.upper()} produced {part_count} usable parts for {mesh_name} "
                             f"with target_parts={target_parts} and max_parts={max_parts}.")


class ConvexDecompositionError(RuntimeError):
    """One or more meshes exceeded the part cap at target_parts=2."""

    def __init__(self, failed_job_ids: Sequence[Hashable], details: str):
        self.failed_job_ids = tuple(failed_job_ids)
        super().__init__(f"Convex decomposition rejected {len(self.failed_job_ids)} candidate mesh(es): {details}")


# ---------------------------------------------------------------------------
# Public decomposer and content addressing
# ---------------------------------------------------------------------------

class ConvexDecomposer:
    """Decompose fingertip meshes with one backend and fixed options, caching results by mesh content."""

    def __init__(self, *, backend: str, max_parts: int, options: dict, workers: int,
                 cache_max_bytes: int, verbose: bool = False):
        if max_parts <= 0:
            raise ValueError("max_parts must be positive")
        self.backend = backend
        self.max_parts = int(max_parts)
        self.options = dict(options)
        self.workers = max(1, int(workers))
        self.cache_max_bytes = int(cache_max_bytes)
        self.verbose = verbose
        self._cache: OrderedDict[str, tuple[ConvexPart, ...]] = OrderedDict()
        self._cache_bytes = 0

    def decompose(self, jobs: Sequence[DecompositionJob]) -> dict[Hashable, list[ConvexPart]]:
        """Convex parts of every job, by ``job_id``. Raises ``ConvexDecompositionError`` if any mesh exceeds the cap."""
        started = time.perf_counter()
        results: dict[Hashable, list[ConvexPart]] = {}
        misses: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        waiters: dict[str, list[Hashable]] = {}
        hits = 0
        for job in jobs:
            vertices = np.ascontiguousarray(job.vertices, dtype=np.float64)
            faces = np.ascontiguousarray(job.faces, dtype=np.int32)
            label = self.backend.upper()
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                raise ValueError(f"{label} vertices for {job.mesh_name} must have shape (N, 3), got {vertices.shape}.")
            if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
                raise ValueError(f"{label} faces for {job.mesh_name} must have shape (M, 3), got {faces.shape}.")
            key = _geometry_key(vertices, faces)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                results[job.job_id] = list(cached)
                hits += 1
                continue
            misses.setdefault(key, (vertices, faces, job.mesh_name))
            waiters.setdefault(key, []).append(job.job_id)

        if misses:
            fresh = self._run(list(misses.values()))
            failed, details = [], []
            for (key, _), parts in zip(misses.items(), fresh):
                if isinstance(parts, ConvexPartCapError):
                    failed.extend(waiters[key])
                    details.append(str(parts))
                    continue
                if not parts:
                    raise RuntimeError(f"{self.backend.upper()} produced no convex parts.")
                if len(parts) > self.max_parts:
                    raise RuntimeError(f"{self.backend.upper()} produced {len(parts)} parts with max_parts={self.max_parts}; "
                                       "refusing to overflow fixed geom slots.")
                self._cache_put(key, parts)
                for job_id in waiters[key]:
                    results[job_id] = list(parts)
            if failed:
                raise ConvexDecompositionError(failed, "; ".join(details))
        if self.verbose:
            backend = self.backend.upper()
            if hits or len(misses) != len(jobs):
                print(f"[{backend}] cache hits: {hits}/{len(jobs)}, unique decompositions: {len(misses)}")
            print(f"[TIMING] {backend} total: {time.perf_counter() - started:.3f}s (cache_hits={hits}, unique_jobs={len(misses)})")
        return results

    def _run(self, meshes: list[tuple[np.ndarray, np.ndarray, str]]) -> list:
        """Parts (or a ConvexPartCapError) of each unique mesh, in order."""
        with _SharedMeshBatch(meshes) as batch:
            if self.backend == "visacd":
                return self._run_visacd(batch)
            if self.backend == "cuacd":
                return self._run_cuacd(batch)
            return self._run_vhacd(batch)

    def _run_visacd(self, batch: _SharedMeshBatch) -> list:
        arrays = _get_visacd_pool().submit(
            _worker_visacd, *batch.shm_args, batch.descriptors, self.options, self.max_parts
        ).result()
        if len(arrays) != len(batch.descriptors) or any(parts is None for parts in arrays):
            raise RuntimeError(f"VisACD returned incomplete results for {len(batch.descriptors)} unique meshes.")
        return [[_frozen_part(v, f) for v, f in parts] for parts in arrays]

    def _run_cuacd(self, batch: _SharedMeshBatch) -> list:
        device_id = int(self.options.get("device_id", 0))
        explicit_pool = self.options.get("pool_bytes")
        pool_bytes = _cuacd_pool_bytes_for_batch(batch) if explicit_pool is None else int(explicit_pool)
        requested_workers = max(1, int(self.options.get("workers", 12)))
        ordered: list = [None] * len(batch.descriptors)
        pending = list(range(len(batch.descriptors)))
        max_rounds = 12
        rounds = 0
        force_serial = False

        def _fail_mesh(idx: int, detail: str) -> None:
            mesh_name = batch.descriptors[idx][4]
            target = int(self.options.get("target_parts", min(8, self.max_parts - 1)))
            ordered[idx] = ConvexPartCapError("cuacd", mesh_name, 0, self.max_parts, target, detail=detail)

        try:
            while pending:
                existing_workers = _cuacd_pool_workers_for(device_id, pool_bytes)
                if existing_workers is not None:
                    max_workers = existing_workers
                else:
                    max_workers = _cuacd_safe_workers(requested_workers, device_id=device_id, pool_bytes=pool_bytes)
                cuacd_workers = 1 if force_serial else max(1, min(len(pending), max_workers))
                pool = _get_cuacd_pool(max_workers, device_id=device_id, pool_bytes=pool_bytes)
                idx_chunks = [pending[i::cuacd_workers] for i in range(cuacd_workers)]
                futures = [
                    pool.submit(
                        _worker_cuacd_chunk, *batch.shm_args,
                        [batch.descriptors[i] for i in chunk], self.options, self.max_parts,
                    )
                    for chunk in idx_chunks
                ]

                next_pending: list[int] = []
                poisoned = pool_broken = False
                chunk_results = []
                for idx_chunk, fut in zip(idx_chunks, futures):
                    try:
                        res_list, failed_local, failed_collateral = fut.result(timeout=30.0)
                    except (concurrent.futures.process.BrokenProcessPool, concurrent.futures.TimeoutError) as exc:
                        poisoned = pool_broken = True
                        chunk_results.append((idx_chunk, None, None, False, exc))
                        continue
                    chunk_results.append((idx_chunk, res_list, failed_local, failed_collateral, None))

                faulting_workers = sum(
                    1 for _, _, failed_local, _, exc in chunk_results
                    if failed_local is not None or exc is not None
                )
                any_primary_fault = any(
                    failed_local is not None and not failed_collateral
                    for _, _, failed_local, failed_collateral, _ in chunk_results
                )
                # When multiple concurrent workers hit a CUDA fault in the same round, sibling
                # contexts may be poisoned mid-kernel; isolate them serially on the next round
                # instead of rejecting every concurrent mesh at once.
                concurrent_ambiguous = (cuacd_workers > 1 and faulting_workers > 1)
                for idx_chunk, res_list, failed_local, failed_collateral, exc in chunk_results:
                    if exc is not None:
                        if len(idx_chunk) == 1 and not concurrent_ambiguous:
                            _fail_mesh(idx_chunk[0], f"worker aborted ({type(exc).__name__})")
                        else:
                            next_pending.extend(idx_chunk)
                        continue
                    for k_i, parts in enumerate(res_list):
                        if parts is None:
                            _fail_mesh(idx_chunk[k_i], f"exceeded max_parts={self.max_parts} after retries")
                        else:
                            ordered[idx_chunk[k_i]] = [_frozen_part(v, f) for v, f in parts]
                    if failed_local is not None:
                        poisoned = True
                        if concurrent_ambiguous or (failed_collateral and (any_primary_fault or len(idx_chunk) > 1)):
                            # Re-run serially after pool reset so only the truly faulting mesh is rejected.
                            next_pending.extend(idx_chunk[failed_local:])
                        else:
                            _fail_mesh(idx_chunk[failed_local], "CUDA kernel fault")
                            next_pending.extend(idx_chunk[failed_local + 1:])
                if poisoned:
                    _reset_cuacd_pool()
                    rounds += 1
                    if concurrent_ambiguous:
                        force_serial = True
                if not next_pending:
                    break
                if (len(next_pending) >= len(pending) and not concurrent_ambiguous) or rounds >= max_rounds:
                    if pool_broken:
                        sys.stderr.write(
                            f"[CUACD] Stopping retries after {rounds} fault round(s); "
                            f"rejecting {len(next_pending)} remaining mesh(es).\n"
                        )
                        sys.stderr.flush()
                    for idx in next_pending:
                        _fail_mesh(idx, "aborted after CuACD worker fault")
                    break
                if pool_broken:
                    sys.stderr.write(f"[CUACD] Worker process aborted; retrying {len(next_pending)} remaining mesh(es).\n")
                    sys.stderr.flush()
                pending = next_pending
        except Exception:
            _reset_cuacd_pool()
            raise

        return ordered

    def _run_vhacd(self, batch: _SharedMeshBatch, *, indices: Sequence[int] | None = None,
                   options: dict | None = None, backend: str | None = None) -> list:
        pool = _get_pool(self.workers)
        opts = self.options if options is None else options
        name = self.backend if backend is None else backend
        selected = range(len(batch.descriptors)) if indices is None else indices
        futures = [
            pool.submit(_worker_decompose, *batch.shm_args, *batch.descriptors[idx], opts, name, self.max_parts)
            for idx in selected
        ]
        out = []
        for future in futures:
            try:
                out.append([_frozen_part(v, f) for v, f in future.result()])
            except ConvexPartCapError as error:
                out.append(error)
        return out

    def _cache_put(self, key: str, parts: Sequence[ConvexPart]) -> None:
        parts = tuple(parts)
        nbytes = sum(p.vertices.nbytes + p.faces.nbytes for p in parts)
        if nbytes > self.cache_max_bytes:
            return
        while self._cache and self._cache_bytes + nbytes > self.cache_max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= sum(p.vertices.nbytes + p.faces.nbytes for p in evicted)
        self._cache[key] = parts
        self._cache_bytes += nbytes


def content_address(world_parts: list[dict[str, list[ConvexPart]]]) -> tuple[list[dict[str, list[ConvexPart]]], int, int]:
    """Name every part ``convex_part_<sha256 of its arrays>`` and share identical ones across worlds.

    Returns the renamed parts and the total and unique part counts.
    """
    canonical: dict[str, ConvexPart] = {}
    by_id: dict[int, ConvexPart] = {}
    renamed, total = [], 0
    for parts_by_tip in world_parts:
        out = {}
        for tip, parts in parts_by_tip.items():
            named = []
            for part in parts:
                pid = id(part)
                cached_part = by_id.get(pid)
                if cached_part is not None:
                    named.append(cached_part)
                    total += 1
                    continue
                digest = hashlib.sha256()
                for array in (part.vertices, part.faces):
                    contiguous = np.ascontiguousarray(array)
                    digest.update(str(contiguous.dtype).encode("ascii"))
                    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
                    digest.update(contiguous.tobytes())
                name = f"convex_part_{digest.hexdigest()}"
                existing = canonical.get(name)
                if existing is None:
                    existing = canonical[name] = replace(part, name=name)
                elif not (np.array_equal(existing.vertices, part.vertices) and np.array_equal(existing.faces, part.faces)):
                    raise RuntimeError(f"Convex-part content digest collision for '{name}'")
                by_id[pid] = existing
                named.append(existing)
                total += 1
            out[tip] = named
        renamed.append(out)
    return renamed, total, len(canonical)


# ---------------------------------------------------------------------------
# Shared-memory batching and caching helpers
# ---------------------------------------------------------------------------

class _SharedMeshBatch:
    """Pack a list of meshes into two shared-memory arrays for worker processes."""

    def __init__(self, meshes: list[tuple[np.ndarray, np.ndarray, str]]):
        self._meshes = meshes
        self.descriptors: list[tuple[int, int, int, int, str]] = []
        self.shm_args: tuple[str, int, str, int] = ("", 0, "", 0)
        self._vertices_shm: shared_memory.SharedMemory | None = None
        self._faces_shm: shared_memory.SharedMemory | None = None

    def __enter__(self) -> _SharedMeshBatch:
        packed_vertices = np.concatenate([v for v, _, _ in self._meshes])
        packed_faces = np.concatenate([f for _, f, _ in self._meshes])
        vertex_offset = face_offset = 0
        for vertices, faces, name in self._meshes:
            self.descriptors.append((vertex_offset, len(vertices), face_offset, len(faces), name))
            vertex_offset += len(vertices)
            face_offset += len(faces)
        self._vertices_shm = _shared_array(packed_vertices)
        try:
            self._faces_shm = _shared_array(packed_faces)
        except Exception:
            try:
                self._vertices_shm.close()
            finally:
                self._vertices_shm.unlink()
            self._vertices_shm = None
            raise
        self.shm_args = (self._vertices_shm.name, len(packed_vertices), self._faces_shm.name, len(packed_faces))
        return self

    def __exit__(self, *_exc) -> None:
        for shm in (self._faces_shm, self._vertices_shm):
            if shm is not None:
                try:
                    shm.close()
                finally:
                    shm.unlink()


def _shared_array(array: np.ndarray) -> shared_memory.SharedMemory:
    array = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)[...] = array
    return shm


def _frozen_part(vertices: np.ndarray, faces: np.ndarray) -> ConvexPart:
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return ConvexPart(vertices=vertices, faces=faces)


def _geometry_key(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (vertices, faces):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Worker pools (one of each per process, reused across envs)
# ---------------------------------------------------------------------------

_POOL: concurrent.futures.ProcessPoolExecutor | None = None
_POOL_WORKERS = 0
_VISACD_POOL: concurrent.futures.ProcessPoolExecutor | None = None
_CUACD_POOL: concurrent.futures.ProcessPoolExecutor | None = None
_CUACD_POOL_KEY: tuple[int, int, int] | None = None
_POOL_LOCK = threading.Lock()
_VISACD_CUDA_PREPROCESS = ("VISACD_ENABLE_CUDA_PREPROCESS", "VISACD_ENABLE_CUDA_PREPROCESS_SIGN", "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE",
                           "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE", "VISACD_ENABLE_CUDA_PREPROCESS_MESH", "VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT")


def _get_pool(workers: int) -> concurrent.futures.ProcessPoolExecutor:
    """A spawned pool of ``workers`` processes; processes start on demand, so a small batch starts few."""
    global _POOL, _POOL_WORKERS
    with _POOL_LOCK:
        if _POOL is None or _POOL_WORKERS != workers:
            if _POOL is not None:
                _POOL.shutdown(wait=False, cancel_futures=True)
            _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            _POOL_WORKERS = workers
        return _POOL


def _get_visacd_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _VISACD_POOL
    with _POOL_LOCK:
        if _VISACD_POOL is None:
            for name in _VISACD_CUDA_PREPROCESS:
                if name == "VISACD_ENABLE_CUDA_PREPROCESS" and os.environ.get("VISACD_DISABLE_CUDA_PREPROCESS", "") not in ("", "0"):
                    continue
                os.environ.setdefault(name, "1")
            _VISACD_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        return _VISACD_POOL


def _cuacd_pool_workers_for(device_id: int, pool_bytes: int) -> int | None:
    """Return the worker count of an already-live CuACD pool for ``(device_id, pool_bytes)``, if any."""
    with _POOL_LOCK:
        if _CUACD_POOL is not None and _CUACD_POOL_KEY is not None:
            w, dev, pb = _CUACD_POOL_KEY
            if dev == int(device_id) and pb == int(pool_bytes):
                return w
    return None


def _get_cuacd_pool(workers: int, device_id: int = 0, pool_bytes: int = 256 * 1024 * 1024) -> concurrent.futures.ProcessPoolExecutor:
    global _CUACD_POOL, _CUACD_POOL_KEY
    key = (int(workers), int(device_id), int(pool_bytes))
    with _POOL_LOCK:
        if _CUACD_POOL is None or _CUACD_POOL_KEY != key:
            if _CUACD_POOL is not None:
                _CUACD_POOL.shutdown(wait=False, cancel_futures=True)
            _CUACD_POOL = concurrent.futures.ProcessPoolExecutor(
                max_workers=key[0],
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_cuacd_worker_init,
                initargs=(key[1], key[2]),
            )
            _CUACD_POOL_KEY = key
        return _CUACD_POOL


def _reset_cuacd_pool() -> None:
    """Shut down and kill the CuACD pool after a worker CUDA context fault or hang."""
    global _CUACD_POOL, _CUACD_POOL_KEY
    with _POOL_LOCK:
        if _CUACD_POOL is not None:
            for proc in list(getattr(_CUACD_POOL, "_processes", {}).values()):
                try:
                    proc.kill()
                except OSError:
                    pass
            _CUACD_POOL.shutdown(wait=False, cancel_futures=True)
            _CUACD_POOL = None
            _CUACD_POOL_KEY = None


def _shutdown_pools() -> None:
    global _POOL, _VISACD_POOL, _CUACD_POOL, _CUACD_POOL_KEY, _POOL_WORKERS
    with _POOL_LOCK:
        for pool in (_POOL, _VISACD_POOL, _CUACD_POOL):
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
        _POOL = _VISACD_POOL = _CUACD_POOL = None
        _CUACD_POOL_KEY = None
        _POOL_WORKERS = 0


atexit.register(_shutdown_pools)


# ---------------------------------------------------------------------------
# Backends (run inside the worker processes)
# ---------------------------------------------------------------------------

def _pop_target_parts(options: dict, *, max_parts: int, backend: str) -> int:
    configured = options.pop("target_parts", None)
    if configured is None:
        if max_parts <= 2:
            raise ValueError("max_parts must be at least 3 so target_parts can remain at least 2 and below the hard part cap.")
        return min(8, max_parts - 1)
    target = int(configured)
    if target < 2:
        raise ValueError(f"{backend.upper()} target_parts must be at least 2, got {target}.")
    if target >= max_parts:
        raise ValueError(f"{backend.upper()} target_parts must be smaller than the hard part cap ({target} >= {max_parts}).")
    return target


def _import_backend(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Convex decomposition backend '{module_name}' is not installed. {install_hint}") from exc


def _run_vhacd(mesh: trimesh.Trimesh, *, max_parts: int, options: dict) -> list[trimesh.Trimesh]:
    if "maxConvexHulls" in options:
        raise ValueError("Use target_parts instead of VHACD's native maxConvexHulls option.")
    options["maxConvexHulls"] = _pop_target_parts(options, max_parts=max_parts, backend="vhacd")
    decomposed = trimesh.decomposition.convex_decomposition(mesh, **options)
    if isinstance(decomposed, trimesh.Trimesh):
        return [decomposed]
    return list(decomposed or [])


_VISACD_CONFIGURABLE = ("return_parts", "score_mode", "flat_surface_min_area", "use_flat_surfaces", "flat_surface_k",
                        "use_merging", "part_limit_policy", "batch_cpu_threads", "max_batch_size",
                        "batch_memory_fraction", "batch_logging")
_VISACD_CONFIG_DEFAULTS: dict[str, object] | None = None


def _run_visacd_batch(meshes: Sequence[tuple[np.ndarray, np.ndarray]], *, max_parts: int, options: Mapping) -> list[list[trimesh.Trimesh]]:
    """Decompose vertex/triangle arrays in one native VisACD batch."""
    global _VISACD_CONFIG_DEFAULTS
    visacd = _import_backend("visacd", "Build it from https://github.com/projectavi/visacd_batched with CUDA and NVIDIA OptiX 8.0 available.")
    if not hasattr(visacd, "process_batch"):
        raise RuntimeError("The installed VisACD does not provide process_batch(). Install https://github.com/projectavi/visacd_batched "
                           "instead of the single-mesh upstream package.")
    missing = [name for name in _VISACD_CONFIGURABLE + ("retain_gpu_resources",) if not hasattr(visacd.config, name)]
    if missing:
        raise RuntimeError(f"The installed VisACD does not expose the required batched configuration controls {missing}. "
                           "Reinstall projectavi/visacd_batched.")
    if not meshes:
        return []
    options = dict(options)
    concavity = float(options.pop("concavity", 0.04))
    if concavity <= 0.0:
        raise ValueError("VisACD concavity must be positive.")
    seed = int(options.pop("seed", 42))
    target = _pop_target_parts(options, max_parts=max_parts, backend="visacd")
    part_limit_policy = str(options.pop("part_limit_policy", "split_budget"))
    if part_limit_policy not in ("split_budget", "adjacent_merge"):
        raise ValueError(f"VisACD part_limit_policy must be 'split_budget' or 'adjacent_merge', got {part_limit_policy!r}.")
    config = {"return_parts": False, "score_mode": str(options.pop("score_mode", "concavity")),
              "use_merging": bool(options.pop("use_merging", False)), "part_limit_policy": part_limit_policy,
              "batch_cpu_threads": int(options.pop("batch_cpu_threads", 0)),
              "max_batch_size": int(options.pop("max_batch_size", 64)),
              "batch_memory_fraction": float(options.pop("batch_memory_fraction", 0.5)),
              "batch_logging": bool(options.pop("batch_logging", False))}
    config.update({name: options.pop(name) for name in ("flat_surface_min_area", "use_flat_surfaces", "flat_surface_k") if name in options})
    retain_gpu_resources = bool(options.pop("retain_gpu_resources", True))
    if options:
        raise ValueError(f"Unknown VisACD option(s): {', '.join(sorted(options))}.")
    if _VISACD_CONFIG_DEFAULTS is None:
        _VISACD_CONFIG_DEFAULTS = {name: getattr(visacd.config, name) for name in _VISACD_CONFIGURABLE}
    for name, value in {**_VISACD_CONFIG_DEFAULTS, **config}.items():
        setattr(visacd.config, name, value)
    visacd.config.retain_gpu_resources = retain_gpu_resources
    visacd.set_seed(seed)
    results = visacd.process_batch(
        [visacd.Mesh(np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(f, dtype=np.int32)) for v, f in meshes],
        concavity=concavity, num_parts=target)
    if len(results) != len(meshes):
        raise RuntimeError(f"VisACD returned {len(results)} results for {len(meshes)} input meshes.")
    batch = []
    for result in results:
        parts = [trimesh.Trimesh(vertices=np.asarray(p.vertices, dtype=np.float64), faces=np.asarray(p.triangles, dtype=np.int64), process=False)
                 for p in result.parts]
        if not parts:
            raise RuntimeError("VisACD produced no convex parts.")
        batch.append(parts)
    return batch


def _clean_parts(decomposed, mesh_name: str, backend: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Process each part, drop degenerate ones (keeping the largest if all are), and thicken flat ones."""
    if isinstance(decomposed, trimesh.Trimesh):
        decomposed = [decomposed]
    processed = []
    for part in decomposed:
        if isinstance(part, dict):
            part = trimesh.Trimesh(vertices=part["vertices"], faces=part["faces"], process=True)
        else:
            part.process()
        processed.append(part)
    if not processed:
        raise RuntimeError(f"{backend.upper()} produced no convex parts for {mesh_name}.")
    kept = [p for p in processed if not _is_thin(p.vertices, p.faces) and abs(float(p.volume)) >= 1e-10]
    if not kept:
        kept = [max(processed, key=lambda p: (abs(float(p.volume)), float(p.extents.prod())))]
    if len(kept) < len(processed):
        sys.stderr.write(f"[{backend.upper()}] Dropped {len(processed) - len(kept)} degenerate part(s) "
                         f"(min bbox dim < {MIN_PART_DIM_M}) for {mesh_name}\n")
        sys.stderr.flush()
    for part in kept:
        flat_axes = (part.vertices.max(axis=0) - part.vertices.min(axis=0)) < 1e-6
        if flat_axes.any():
            vertices = part.vertices.copy()
            vertices[0, flat_axes] += 1e-4
            part.vertices = vertices
    return [(np.ascontiguousarray(p.vertices, dtype=np.float64), np.ascontiguousarray(p.faces, dtype=np.int32)) for p in kept]


def _is_thin(vertices, faces) -> bool:
    if len(vertices) < 4 or len(faces) < 4:
        return True
    vertices = np.asarray(vertices)
    return np.min(vertices.max(axis=0) - vertices.min(axis=0)) < MIN_PART_DIM_M


def _decompose_once(mesh: trimesh.Trimesh, backend: str, max_parts: int, options: dict) -> list:
    options = dict(options)
    if backend == "vhacd":
        parts = _run_vhacd(mesh, max_parts=max_parts, options=options)
    else:
        raise ValueError(f"Unsupported single-mesh backend '{backend}'.")
    if not parts:
        raise RuntimeError(f"{backend.upper()} produced no convex parts.")
    return parts


def _decompose_with_retries(vertices, faces, mesh_name: str, options: dict, backend: str, max_parts: int) -> list[tuple[np.ndarray, np.ndarray]]:
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=False)
    target = int(options["target_parts"]) if "target_parts" in options else min(8, max_parts - 1)
    while True:
        parts = _clean_parts(_decompose_once(mesh, backend, max_parts, {**options, "target_parts": target}), mesh_name, backend)
        if len(parts) <= max_parts:
            return parts
        if target <= 2:
            raise ConvexPartCapError(backend, mesh_name, len(parts), max_parts, target)
        target = max(2, target // 2)
        sys.stderr.write(f"[{backend.upper()}] Retrying {mesh_name} with target_parts={target}.\n")
        sys.stderr.flush()


def _release_freed_memory() -> None:
    """Hand memory freed by the backend back to the OS (glibc only). VHACD's working memory
    otherwise stays with the worker: about 0.5 GB after one job, growing with every job."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _attach(vertices_shm_name: str, total_vertices: int, faces_shm_name: str, total_faces: int):
    vertices_shm = shared_memory.SharedMemory(name=vertices_shm_name)
    try:
        faces_shm = shared_memory.SharedMemory(name=faces_shm_name)
    except Exception:
        vertices_shm.close()
        raise
    vertices = np.ndarray((total_vertices, 3), dtype=np.float64, buffer=vertices_shm.buf)
    faces = np.ndarray((total_faces, 3), dtype=np.int32, buffer=faces_shm.buf)
    return vertices_shm, faces_shm, vertices, faces


def _worker_decompose(vertices_shm_name, total_vertices, faces_shm_name, total_faces,
                      vertex_offset, vertex_count, face_offset, face_count, mesh_name, options, backend, max_parts):
    """Pool task: one VHACD job read from the packed shared-memory arrays."""
    vertices_shm, faces_shm, vertices, faces = _attach(vertices_shm_name, total_vertices, faces_shm_name, total_faces)
    try:
        return _decompose_with_retries(vertices[vertex_offset:vertex_offset + vertex_count], faces[face_offset:face_offset + face_count],
                                       mesh_name, options, backend, max_parts)
    finally:
        faces_shm.close()
        vertices_shm.close()
        _release_freed_memory()


def _worker_visacd(vertices_shm_name, total_vertices, faces_shm_name, total_faces, descriptors, options, max_parts):
    """Pool task: every VisACD job as one batch, retrying over-cap meshes with halved targets, then VHACD."""
    vertices_shm, faces_shm, vertices, faces = _attach(vertices_shm_name, total_vertices, faces_shm_name, total_faces)
    results: list = [None] * len(descriptors)
    try:
        meshes = [(vertices[vo:vo + vc], faces[fo:fo + fc]) for vo, vc, fo, fc, _ in descriptors]
        original_target = target = int(options["target_parts"]) if "target_parts" in options else min(8, max_parts - 1)
        pending = list(range(len(descriptors)))
        while pending:
            batch = _run_visacd_batch([meshes[i] for i in pending], max_parts=max_parts, options={**options, "target_parts": target})
            retry = []
            for index, decomposed in zip(pending, batch):
                parts = _clean_parts(decomposed, descriptors[index][4], "visacd")
                if len(parts) > max_parts:
                    retry.append((index, len(parts)))
                else:
                    results[index] = parts
            if not retry:
                break
            if target <= 2:
                for index, count in retry:
                    name = descriptors[index][4]
                    sys.stderr.write(f"[VISACD] Exhausted targets for {name} ({count} > max_parts={max_parts}); "
                                     f"falling back to VHACD at target_parts={original_target}.\n")
                    sys.stderr.flush()
                    mesh = trimesh.Trimesh(vertices=np.asarray(meshes[index][0], dtype=np.float64),
                                           faces=np.asarray(meshes[index][1], dtype=np.int64), process=False)
                    results[index] = _clean_parts(_decompose_once(mesh, "vhacd", max_parts, {"target_parts": original_target}), name, "vhacd")
                break
            target = max(2, target // 2)
            sys.stderr.write(f"[VISACD] Retrying {len(retry)} mesh(es) with target_parts={target}.\n")
            sys.stderr.flush()
            pending = [index for index, _ in retry]
    finally:
        faces_shm.close()
        vertices_shm.close()
        _release_freed_memory()
    return results


def _cuacd_pool_bytes_for_batch(batch: _SharedMeshBatch) -> int:
    """Choose a power-of-2 CuACD scratch pool size from the largest mesh face count in the batch."""
    max_faces = max((int(desc[3]) for desc in batch.descriptors), default=0)
    if max_faces <= 10_000:
        return 512 * 1024 * 1024
    return 1024 * 1024 * 1024


def _cuacd_safe_workers(requested: int, *, device_id: int, pool_bytes: int, reserve_bytes: int = 4 * 1024 ** 3) -> int:
    """Cap CuACD worker count to fit comfortably inside free GPU VRAM (target = requested, e.g. 12)."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info(device_id)
            per_worker = int(pool_bytes) + 450 * 1024 * 1024  # scratch pool + CUDA context overhead
            affordable = max(1, int(max(0, free_bytes - reserve_bytes) // per_worker))
            return max(1, min(requested, affordable))
    except Exception:
        pass
    return max(1, requested)


_CUACD_CTX = None


def _cuacd_worker_init(device_id: int, pool_bytes: int) -> None:
    """Initialize one persistent CuACD GPU context per spawned worker process."""
    global _CUACD_CTX
    if sys.platform.startswith("linux"):
        try:
            import signal
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: reap worker if parent dies
        except Exception:
            pass
    cuacd = _import_backend("cuacd", "Install eliphatfs/cuacd (built for your GPU arch).")
    _CUACD_CTX = cuacd.Context(device=device_id, pool_bytes=pool_bytes)


def _worker_cuacd_chunk(vertices_shm_name, total_vertices, faces_shm_name, total_faces, descriptors, options, max_parts):
    """Pool task: decompose a chunk of meshes on GPU with CuACD using automatic [-1, 1] normalization.

    Returns ``(completed_parts, failed_local_index)``. On a CUDA kernel fault (e.g. error 719),
    stops immediately so the parent can recycle the poisoned worker pool and fall back to VHACD
    for only the single faulting mesh.
    """
    vertices_shm, faces_shm, vertices, faces = _attach(vertices_shm_name, total_vertices, faces_shm_name, total_faces)
    results: list = []
    failed_local: int | None = None
    failed_collateral = False
    threshold_base = float(options.get("threshold", 0.02))
    depth = int(options.get("depth", 2))
    width = int(options.get("width", 60))
    merge_hulls = bool(options.get("merge_hulls", False))
    n_concave_edges = int(options.get("n_concave_edges", 0))
    original_target = int(options["target_parts"]) if "target_parts" in options else min(8, max_parts - 1)
    try:
        for local_i, (vo, vc, fo, fc, mesh_name) in enumerate(descriptors):
            v = np.ascontiguousarray(vertices[vo:vo + vc], dtype=np.float32)
            f = np.ascontiguousarray(faces[fo:fo + fc], dtype=np.int32)
            lo, hi = v.min(axis=0), v.max(axis=0)
            center = 0.5 * (lo + hi)
            extent = float(np.max(hi - lo))
            scale = 0.5 * extent if extent > 0.0 else 1.0
            v_norm = np.ascontiguousarray((v - center) / scale, dtype=np.float32)

            thresh = threshold_base
            parts_clean = None
            try:
                for _ in range(4):
                    raw = _CUACD_CTX.lookahead_decompose(
                        v_norm, f, threshold=thresh, depth=depth, width=width,
                        merge_hulls=merge_hulls, n_concave_edges=n_concave_edges,
                    )
                    trimeshes = [
                        trimesh.Trimesh(
                            vertices=np.asarray(hv, dtype=np.float64) * scale + center.astype(np.float64),
                            faces=np.asarray(hf, dtype=np.int64),
                            process=False,
                        )
                        for _, _, hv, hf in raw
                    ]
                    cand = _clean_parts(trimeshes, mesh_name, "cuacd")
                    _CUACD_CTX.heap_compact()
                    if 0 < len(cand) <= max_parts:
                        parts_clean = cand
                        break
                    thresh *= 2.0
            except RuntimeError as exc:
                failed_collateral = "cuStreamCreate" in str(exc)
                if failed_collateral:
                    sys.stderr.write(f"[CUACD] Collateral context fault on {mesh_name} ({exc}); retrying after pool reset.\n")
                else:
                    sys.stderr.write(f"[CUACD] Kernel fault on {mesh_name} ({exc}); rejecting this mesh.\n")
                sys.stderr.flush()
                failed_local = local_i
                break
            results.append(parts_clean)
    finally:
        faces_shm.close()
        vertices_shm.close()
    return results, failed_local, failed_collateral
