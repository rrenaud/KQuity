#!/usr/bin/env python3
"""Memory and time benchmarking utilities."""

import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB (works on Linux).

    Note: On Linux, ru_maxrss is in KB. On macOS, it's in bytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux returns KB, convert to MB
    return usage.ru_maxrss / 1024


def get_current_memory_mb() -> float:
    """Get current memory usage in MB using psutil if available, else None."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    elapsed_seconds: float
    peak_memory_mb: float
    current_memory_mb: float

    def __str__(self) -> str:
        return (f"{self.name}: {self.elapsed_seconds:.1f}s, "
                f"peak {self.peak_memory_mb:.0f} MB, "
                f"current {self.current_memory_mb:.0f} MB")


class MemoryBenchmark:
    """Context manager for benchmarking time and memory usage.

    Usage:
        with MemoryBenchmark("my operation") as bench:
            # do work
        print(bench.result)
    """

    def __init__(self, name: str):
        self.name = name
        self.result: Optional[BenchmarkResult] = None

    def __enter__(self) -> 'MemoryBenchmark':
        self.start_time = time.time()
        self.start_peak = get_peak_memory_mb()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.time() - self.start_time
        peak_mem = get_peak_memory_mb()
        current_mem = get_current_memory_mb()

        self.result = BenchmarkResult(
            name=self.name,
            elapsed_seconds=elapsed,
            peak_memory_mb=peak_mem,
            current_memory_mb=current_mem,
        )
        print(self.result)


@contextmanager
def benchmark(name: str):
    """Simple context manager for quick benchmarking.

    Usage:
        with benchmark("loading data"):
            data = load_data()
    """
    start_time = time.time()
    start_peak = get_peak_memory_mb()

    yield

    elapsed = time.time() - start_time
    peak_mem = get_peak_memory_mb()
    print(f"{name}: {elapsed:.1f}s, peak {peak_mem:.0f} MB")


def print_memory_stats():
    """Print current memory statistics."""
    peak = get_peak_memory_mb()
    current = get_current_memory_mb()
    print(f"Memory: current {current:.0f} MB, peak {peak:.0f} MB")


if __name__ == '__main__':
    # Quick test
    import numpy as np

    print("Testing memory benchmark utilities...")
    print_memory_stats()

    with MemoryBenchmark("allocate 100MB array") as bench:
        arr = np.zeros((100, 1024, 1024), dtype=np.float64)  # ~800 MB

    print(f"\nArray shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Array size: {arr.nbytes / 1024 / 1024:.1f} MB")
    print_memory_stats()
