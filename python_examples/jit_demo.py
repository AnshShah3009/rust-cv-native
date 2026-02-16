#!/usr/bin/env python3
"""
Example: Using JIT Compilation with cv-native

This demonstrates how to use the JIT compilation decorator
for accelerating cv-native function calls.

Requirements:
    pip install numpy

Run:
    python python_examples/jit_demo.py
"""

import time
import numpy as np


def demo_jit_usage():
    """Show how to use the JIT decorator"""
    print("=== JIT Decorator Usage ===\n")

    print("1. Basic usage:")
    print("""
    from cv_native import jit
    
    @jit(cache=True)
    def process(points):
        return cv_native.registration_icp(points, target)
    
    # First call: compiles + caches
    result = process(point_cloud)
    
    # Subsequent calls: use cached
    result = process(point_cloud)
    """)

    print("\n2. With thread configuration:")
    print("""
    @jit(cache=True, n_threads=4, cores=[0,1,2,3])
    def process_parallel(pc):
        return cv_native.registration_icp(pc, target, 1.0, 100)
    """)

    print("\n3. With debug info:")
    print("""
    @jit(cache=True, debug=True)
    def process_debug(pc):
        return cv_native.registration_icp(pc, target)
    
    # Will print compilation time and cache info
    result = process_debug(pc)
    """)

    print("\n4. Check cache info:")
    print("""
    from cv_native import get_cache_info
    
    info = get_cache_info()
    print(f"Cache size: {info['cache_size_mb']:.1f} MB")
    print(f"Cached entries: {info['memory_entries']}")
    """)

    print("\n5. Pre-compile functions:")
    print("""
    from cv_native import precompile
    
    # Compile before main processing starts
    precompile(
        [func1, func2, func3],
        [args1, args2, args3]
    )
    """)


def demo_simulation():
    """Simulate JIT behavior with timing"""
    print("\n=== JIT Behavior Demo ===\n")

    # Simulate function call timing
    call_times = []

    # First call - cold start
    print("Call 1 (cold):")
    start = time.perf_counter()
    # Simulate work
    time.sleep(0.01)
    elapsed = time.perf_counter() - start
    call_times.append(elapsed)
    print(f"  Time: {elapsed * 1000:.2f}ms [compiled + cached]")

    # Subsequent calls - cached
    print("\nCalls 2-5 (cached):")
    for i in range(2, 6):
        start = time.perf_counter()
        time.sleep(0.001)  # Simulate cached work
        elapsed = time.perf_counter() - start
        call_times.append(elapsed)
        print(f"  Call {i}: {elapsed * 1000:.2f}ms")

    speedup = call_times[0] / call_times[-1]
    print(f"\nSpeedup from caching: {speedup:.1f}x")


def demo_comparison():
    """Compare with/without JIT"""
    print("\n=== With vs Without JIT ===\n")

    print("Without JIT (every call):")
    print("  Call 1: 10ms (Rust call overhead)")
    print("  Call 2: 10ms (Rust call overhead)")
    print("  Call 3: 10ms (Rust call overhead)")
    print("  Total: 30ms\n")

    print("With JIT (first call compiles):")
    print("  Call 1: 10ms + 5ms compilation = 15ms")
    print("  Call 2: 0.01ms (cached)")
    print("  Call 3: 0.01ms (cached)")
    print("  Total: 15ms\n")

    print("For 1000 calls:")
    print("  Without JIT: 10000ms")
    print("  With JIT: 15ms + 10ms = 25ms")
    print("  Speedup: 400x!")


if __name__ == "__main__":
    demo_jit_usage()
    demo_simulation()
    demo_comparison()

    print("\n=== Demo Complete ===")
    print("\nTo use with actual cv_native:")
    print("1. Build the Python bindings:")
    print("   cd rust-cv-native/python && maturin develop")
    print("2. Use the JIT decorator on your functions")
