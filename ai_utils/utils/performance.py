"""
Performance benchmarking utilities for AI operations.
"""

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""

    operation: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    total_items: int
    items_per_second: float
    success_rate: float
    error_count: int
    metadata: dict[str, Any]


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, label: str = None):
        self.label = label
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


class PerformanceBenchmark:
    """Performance benchmarking utility"""

    def __init__(self, label: str = None):
        self.label = label
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.results: list[BenchmarkResult] = []
        self.measurements: dict[str, list[float]] = {}

    def measure(self, operation_name: str = "operation"):
        """Get a context manager for timing operations"""
        return PerformanceTimer(operation_name)

    def record_measurement(self, operation_name: str, elapsed_ms: float):
        """Record a measurement"""
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        self.measurements[operation_name].append(elapsed_ms)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all measurements"""
        stats = {}
        for operation, times in self.measurements.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "avg_ms": statistics.mean(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "median_ms": statistics.median(times),
                    "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
                    "total_ms": sum(times),
                }
        return stats

    async def benchmark_embedding_operation(
        self,
        operation: Callable,
        test_data: list[str],
        operation_name: str = "embedding",
        warmup_runs: int = 3,
        metadata: dict[str, Any] = None,
    ) -> BenchmarkResult:
        """
        Benchmark an embedding operation.

        Args:
            operation: Async function to benchmark
            test_data: List of test inputs
            operation_name: Name of the operation
            warmup_runs: Number of warmup runs
            metadata: Additional metadata

        Returns:
            Benchmark result
        """
        times = []
        errors = 0
        total_items = len(test_data)

        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup runs for {operation_name}")
        for _ in range(warmup_runs):
            try:
                start_time = time.time()
                await operation(test_data[0] if test_data else "")
                warmup_time = time.time() - start_time
                logger.debug(f"Warmup run took {warmup_time:.3f}s")
            except Exception as e:
                logger.warning(f"Warmup run failed: {str(e)}")

        # Actual benchmark runs
        logger.info(f"Running benchmark for {operation_name} with {total_items} items")
        for i, item in enumerate(test_data):
            try:
                start_time = time.time()
                await operation(item)
                end_time = time.time()
                times.append(end_time - start_time)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{total_items} items")

            except Exception as e:
                errors += 1
                logger.warning(f"Operation failed for item {i}: {str(e)}")

        # Calculate statistics
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            median_time = statistics.median(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            items_per_second = len(times) / sum(times)
        else:
            avg_time = min_time = max_time = median_time = std_dev = (
                items_per_second
            ) = 0

        success_rate = (len(times) / total_items) * 100 if total_items > 0 else 0

        result = BenchmarkResult(
            operation=operation_name,
            total_time=sum(times),
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            total_items=total_items,
            items_per_second=items_per_second,
            success_rate=success_rate,
            error_count=errors,
            metadata=metadata or {},
        )

        self.results.append(result)
        return result

    async def benchmark_batch_operation(
        self,
        operation: Callable,
        test_data: list[list[str]],
        operation_name: str = "batch_embedding",
        warmup_runs: int = 2,
        metadata: dict[str, Any] = None,
    ) -> BenchmarkResult:
        """
        Benchmark a batch operation.

        Args:
            operation: Async function to benchmark
            test_data: List of test batches
            operation_name: Name of the operation
            warmup_runs: Number of warmup runs
            metadata: Additional metadata

        Returns:
            Benchmark result
        """
        times = []
        errors = 0
        total_items = sum(len(batch) for batch in test_data)

        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup runs for {operation_name}")
        for _ in range(warmup_runs):
            try:
                if test_data:
                    start_time = time.time()
                    await operation(test_data[0])
                    warmup_time = time.time() - start_time
                    logger.debug(f"Warmup run took {warmup_time:.3f}s")
            except Exception as e:
                logger.warning(f"Warmup run failed: {str(e)}")

        # Actual benchmark runs
        logger.info(
            f"Running benchmark for {operation_name} with {len(test_data)} batches"
        )
        for i, batch in enumerate(test_data):
            try:
                start_time = time.time()
                await operation(batch)
                end_time = time.time()
                times.append(end_time - start_time)

                logger.debug(
                    f"Processed batch {i + 1}/{len(test_data)} ({len(batch)} items)"
                )

            except Exception as e:
                errors += 1
                logger.warning(f"Batch operation failed for batch {i}: {str(e)}")

        # Calculate statistics
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            median_time = statistics.median(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            items_per_second = total_items / sum(times)
        else:
            avg_time = min_time = max_time = median_time = std_dev = (
                items_per_second
            ) = 0

        success_rate = (len(times) / len(test_data)) * 100 if test_data else 0

        result = BenchmarkResult(
            operation=operation_name,
            total_time=sum(times),
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            total_items=total_items,
            items_per_second=items_per_second,
            success_rate=success_rate,
            error_count=errors,
            metadata=metadata or {},
        )

        self.results.append(result)
        return result

    def generate_report(self) -> str:
        """
        Generate a performance report.

        Returns:
            Formatted performance report
        """
        if not self.results:
            return "No benchmark results available."

        report = ["# Performance Benchmark Report", ""]

        for result in self.results:
            report.append(f"## {result.operation}")
            report.append("")
            report.append(f"- **Total Time**: {result.total_time:.3f}s")
            report.append(f"- **Average Time**: {result.avg_time:.3f}s")
            report.append(f"- **Min Time**: {result.min_time:.3f}s")
            report.append(f"- **Max Time**: {result.max_time:.3f}s")
            report.append(f"- **Median Time**: {result.median_time:.3f}s")
            report.append(f"- **Standard Deviation**: {result.std_dev:.3f}s")
            report.append(f"- **Total Items**: {result.total_items}")
            report.append(f"- **Items/Second**: {result.items_per_second:.2f}")
            report.append(f"- **Success Rate**: {result.success_rate:.1f}%")
            report.append(f"- **Error Count**: {result.error_count}")

            if result.metadata:
                report.append("- **Metadata**:")
                for key, value in result.metadata.items():
                    report.append(f"  - {key}: {value}")

            report.append("")

        return "\n".join(report)

    def get_summary_stats(self) -> dict[str, Any]:
        """
        Get summary statistics across all benchmarks.

        Returns:
            Summary statistics
        """
        if not self.results:
            return {}

        total_operations = len(self.results)
        total_items = sum(r.total_items for r in self.results)
        total_time = sum(r.total_time for r in self.results)
        avg_success_rate = statistics.mean(r.success_rate for r in self.results)
        total_errors = sum(r.error_count for r in self.results)

        return {
            "total_operations": total_operations,
            "total_items_processed": total_items,
            "total_time": total_time,
            "average_success_rate": avg_success_rate,
            "total_errors": total_errors,
            "operations": [r.operation for r in self.results],
        }

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.label:
            print(f"[PERF] Starting: {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = (self.end_time - self.start_time) * 1000
        if self.label:
            print(f"[PERF] Finished: {self.label} in {self.elapsed_time:.2f}ms")


async def benchmark_embedding_service(
    service, test_texts: list[str], batch_sizes: list[int] = None
) -> PerformanceBenchmark:
    """
    Benchmark an embedding service with various configurations.

    Args:
        service: Embedding service to benchmark
        test_texts: List of test texts
        batch_sizes: List of batch sizes to test

    Returns:
        Performance benchmark with results
    """
    benchmark = PerformanceBenchmark()

    if batch_sizes is None:
        batch_sizes = [1, 5, 10, 20, 50]

    # Benchmark single text embedding
    logger.info("Benchmarking single text embedding")
    await benchmark.benchmark_embedding_operation(
        operation=lambda text: service.embed_text(text),
        test_data=test_texts[:50],  # Limit for single operations
        operation_name="single_text_embedding",
        metadata={
            "model": getattr(service.config, "openai", {}).get("model", "unknown")
        },
    )

    # Benchmark batch operations with different sizes
    for batch_size in batch_sizes:
        if batch_size > len(test_texts):
            continue

        # Create batches
        batches = [
            test_texts[i : i + batch_size]
            for i in range(0, len(test_texts), batch_size)
        ]

        logger.info(f"Benchmarking batch embedding with batch size {batch_size}")
        await benchmark.benchmark_batch_operation(
            operation=lambda texts: service.embed_batch(texts),
            test_data=batches,
            operation_name=f"batch_embedding_{batch_size}",
            metadata={
                "batch_size": batch_size,
                "model": getattr(service.config, "openai", {}).get("model", "unknown"),
            },
        )

    return benchmark
