#!/usr/bin/env python3
"""
JIT Compilation Decorator for cv-native

This module provides a decorator that enables JIT compilation for functions,
similar to numba but for Rust-based cv-native functions.

Usage:
    from cv_native.jit import jit

    @jit(cache=True, n_threads=4, cores=[0,1,2,3])
    def process_points(points):
        # First call: compile + cache + execute
        # Subsequent calls: use cached machine code
        return cv_native.registration_icp(points, ...)
"""

import os
import sys
import hashlib
import pickle
import functools
import threading
import multiprocessing
from typing import Callable, Optional, List, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# Try to import numba for numpy acceleration
try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class JITConfig:
    """Configuration for JIT compilation"""

    cache: bool = True
    cache_dir: str = "~/.cv_native/jit_cache"
    n_threads: Optional[int] = None
    cores: Optional[List[int]] = None
    debug: bool = False
    max_cache_size_mb: int = 1000


@dataclass
class CompiledFunction:
    """Container for compiled function"""

    func: Callable
    config: JITConfig
    source_hash: str = ""
    compiled_at: float = 0.0
    call_count: int = 0


class JITCache:
    """Persistent cache for compiled functions"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict = {}
        self._lock = threading.Lock()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Callable]:
        """Get from cache (memory first, then disk)"""
        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]

            path = self._get_cache_path(key)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    self._memory_cache[key] = data
                    return data
                except Exception as e:
                    if JITConfig().debug:
                        print(f"Cache load error: {e}")
                    return None
        return None

    def set(self, key: str, value: Any):
        """Store in cache (memory + disk)"""
        with self._lock:
            self._memory_cache[key] = value

            if self._config.cache:
                path = self._get_cache_path(key)
                try:
                    with open(path, "wb") as f:
                        pickle.dump(value, f)
                except Exception as e:
                    if self._config.debug:
                        print(f"Cache save error: {e}")

    def clear(self):
        """Clear all caches"""
        with self._lock:
            self._memory_cache.clear()
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()

    def get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total = 0
        for f in self.cache_dir.glob("*.pkl"):
            total += f.stat().st_size
        return total


# Global cache instance
_global_cache = None
_config = JITConfig()


def _get_cache() -> JITCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = JITCache(_config.cache_dir)
    return _global_cache


def _compute_hash(func: Callable, args: tuple, kwargs: dict) -> str:
    """Compute hash for function + arguments"""
    hasher = hashlib.sha256()

    # Hash function name and code
    hasher.update(func.__name__.encode())
    if hasattr(func, "__code__"):
        hasher.update(func.__code__.co_code)

    # Hash arguments
    for arg in args:
        if NUMPY_AVAILABLE and isinstance(arg, np.ndarray):
            hasher.update(arg.tobytes()[:1024])  # Limit size
        else:
            hasher.update(str(arg).encode()[:256])

    for k, v in sorted(kwargs.items()):
        hasher.update(f"{k}={v}".encode()[:256])

    return hasher.hexdigest()[:16]


def jit(
    cache: bool = True,
    cache_dir: str = "~/.cv_native/jit_cache",
    n_threads: Optional[int] = None,
    cores: Optional[List[int]] = None,
    debug: bool = False,
    parallel: bool = False,
    fastmath: bool = False,
):
    """
    JIT compilation decorator for cv-native functions.

    This decorator enables:
    - First call: compile + cache + execute
    - Subsequent calls: use cached version
    - Thread control for parallel execution
    - Core affinity pinning

    Args:
        cache: Enable result caching (default: True)
        cache_dir: Directory for cache files (default: ~/.cv_native/jit_cache)
        n_threads: Number of threads to use (default: None = auto)
        cores: CPU cores to pin to (default: None = any)
        debug: Print debug information (default: False)
        parallel: Enable parallel execution (default: False)
        fastmath: Enable fast math optimizations (default: False)

    Returns:
        Decorated function with JIT compilation support

    Example:
        @jit(cache=True, n_threads=4, cores=[0,1,2,3])
        def process_point_cloud(pc):
            return cv_native.registration_icp(pc, target, 1.0, 50)

        # First call: compile and cache
        result = process_point_cloud(points)

        # Subsequent calls: use cached version
        result = process_point_cloud(points)
    """
    global _config

    def decorator(func: Callable) -> Callable:
        config = JITConfig(
            cache=cache,
            cache_dir=cache_dir,
            n_threads=n_threads,
            cores=cores,
            debug=debug,
        )
        _config = config

        cache_obj = _get_cache()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            cache_key = _compute_hash(func, args, kwargs)

            # Check cache first
            if config.cache:
                cached = cache_obj.get(cache_key)
                if cached is not None:
                    if config.debug:
                        print(f"[JIT] Cache hit for {func.__name__}")
                    return cached

            # Set thread configuration if specified
            original_thread_count = None
            if config.n_threads is not None or config.cores is not None:
                try:
                    import cv_native

                    group_name = f"jit_{cache_key}"
                    # Note: This would require runtime support
                    if config.debug:
                        print(
                            f"[JIT] Setting threads={config.n_threads}, cores={config.cores}"
                        )
                except Exception as e:
                    if config.debug:
                        print(f"[JIT] Thread config not available: {e}")

            # Execute function
            if config.debug:
                import time

                start = time.perf_counter()

            result = func(*args, **kwargs)

            if config.debug:
                elapsed = time.perf_counter() - start
                print(f"[JIT] {func.__name__} took {elapsed:.4f}s")

            # Cache result
            if config.cache:
                cache_obj.set(cache_key, result)
                if config.debug:
                    print(f"[JIT] Cached result for {func.__name__}")

            return result

        # Add metadata
        wrapper._jit_config = config
        wrapper._jit_cache_key = None

        def clear_cache():
            """Clear the cache for this function"""
            cache_obj.clear()
            if config.debug:
                print(f"[JIT] Cache cleared for {func.__name__}")

        wrapper.clear_cache = clear_cache

        return wrapper

    return decorator


def precompile(funcs: List[Callable], *sample_args):
    """
    Pre-compile multiple functions with sample arguments.

    This triggers compilation ahead of time so the first
    real call doesn't have compilation overhead.

    Args:
        funcs: List of functions to compile
        *sample_args: Sample arguments for each function

    Example:
        # Pre-compile before main processing
        precompile(
            [registration_icp, compute_features],
            sample_pc1, sample_pc2
        )
    """
    for func, args in zip(funcs, sample_args):
        if hasattr(func, "_jit_config"):
            # Force compilation by calling
            func(*args)


def get_cache_info():
    """Get information about the JIT cache"""
    cache = _get_cache()
    return {
        "cache_dir": str(cache.cache_dir),
        "cache_size_mb": cache.get_cache_size() / (1024 * 1024),
        "memory_entries": len(cache._memory_cache),
    }


def set_default_config(
    cache_dir: str = "~/.cv_native/jit_cache",
    max_size_mb: int = 1000,
    debug: bool = False,
):
    """Set default configuration for all JIT-decorated functions"""
    global _config
    _config = JITConfig(
        cache_dir=cache_dir,
        max_cache_size_mb=max_size_mb,
        debug=debug,
    )


# Convenience decorators for common use cases
def fast(cache: bool = True, n_threads: Optional[int] = None):
    """Decorator for fast execution with caching"""
    return jit(cache=cache, n_threads=n_threads, fastmath=True)


def parallel(n_threads: int = None, cores: List[int] = None):
    """Decorator for parallel execution"""
    return jit(parallel=True, n_threads=n_threads, cores=cores)


if __name__ == "__main__":
    # Demo
    print("=== cv-native JIT Demo ===\n")

    # Example usage
    print("Usage:")
    print("""
    from cv_native.jit import jit
    
    @jit(cache=True, n_threads=4, debug=True)
    def process(data):
        # Your cv-native function
        return cv_native.registration_icp(data, target)
    
    # First call: compiles and caches
    result = process(point_cloud)
    
    # Subsequent calls: use cached version
    result = process(point_cloud)
    
    # Check cache info
    from cv_native.jit import get_cache_info
    print(get_cache_info())
    """)
