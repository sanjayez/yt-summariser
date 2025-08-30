"""
System metrics collection and monitoring utilities.

This module provides functions for collecting and logging system-level
statistics including CPU, memory, disk, and network usage.
"""

import os
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

from ..logging import get_logger

# Default logger for system metrics
logger = get_logger(__name__)


def log_system_stats(
    logger_instance: Any | None = None,
    include_disk: bool = True,
    include_network: bool = False,
) -> dict[str, Any]:
    """
    Log comprehensive system statistics.

    Args:
        logger_instance: Logger to use (defaults to module logger)
        include_disk: Whether to include disk I/O statistics
        include_network: Whether to include network I/O statistics

    Returns:
        Dictionary containing system statistics

    Example:
        >>> stats = log_system_stats()
        >>> # Logs CPU, memory, and optionally disk/network statistics
    """
    log = logger_instance or logger
    stats = {}

    if not HAS_PSUTIL:
        log.warning("psutil not available, system statistics disabled")
        return {}

    # CPU statistics
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    stats["cpu"] = {
        "total_percent": sum(cpu_percent) / len(cpu_percent),
        "per_core": cpu_percent,
        "count": psutil.cpu_count(),
        "count_physical": psutil.cpu_count(logical=False),
    }

    # Memory statistics
    virtual_memory = psutil.virtual_memory()
    stats["memory"] = {
        "total_mb": virtual_memory.total / 1024 / 1024,
        "available_mb": virtual_memory.available / 1024 / 1024,
        "used_mb": virtual_memory.used / 1024 / 1024,
        "percent": virtual_memory.percent,
    }

    # Process-specific statistics
    process = psutil.Process(os.getpid())
    stats["process"] = {
        "num_threads": process.num_threads(),
        "num_fds": process.num_fds() if hasattr(process, "num_fds") else None,
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
    }

    log.info(
        f"System Stats: "
        f"CPU={stats['cpu']['total_percent']:.1f}% "
        f"({stats['cpu']['count']} cores), "
        f"Memory={stats['memory']['percent']:.1f}% "
        f"({stats['memory']['used_mb']:.0f}/{stats['memory']['total_mb']:.0f}MB), "
        f"Process: {stats['process']['num_threads']} threads, "
        f"{stats['process']['memory_mb']:.0f}MB"
    )

    # Disk I/O statistics
    if include_disk:
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                stats["disk_io"] = {
                    "read_mb": disk_io.read_bytes / 1024 / 1024,
                    "write_mb": disk_io.write_bytes / 1024 / 1024,
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count,
                }
                log.debug(
                    f"  Disk I/O: "
                    f"Read={stats['disk_io']['read_mb']:.2f}MB, "
                    f"Write={stats['disk_io']['write_mb']:.2f}MB"
                )
        except Exception as e:
            log.debug(f"Could not collect disk I/O stats: {e}")

    # Network I/O statistics
    if include_network:
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                stats["network_io"] = {
                    "sent_mb": net_io.bytes_sent / 1024 / 1024,
                    "recv_mb": net_io.bytes_recv / 1024 / 1024,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                log.debug(
                    f"  Network I/O: "
                    f"Sent={stats['network_io']['sent_mb']:.2f}MB, "
                    f"Received={stats['network_io']['recv_mb']:.2f}MB"
                )
        except Exception as e:
            log.debug(f"Could not collect network I/O stats: {e}")

    return stats


def get_process_info() -> dict[str, Any]:
    """
    Get detailed information about the current process.

    Returns:
        Dictionary containing process information
    """
    if not HAS_PSUTIL:
        return {"error": "psutil not available"}

    from datetime import datetime

    process = psutil.Process(os.getpid())

    info = {
        "pid": process.pid,
        "name": process.name(),
        "status": process.status(),
        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
        "num_threads": process.num_threads(),
        "memory_info": {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
        },
        "cpu_times": {
            "user": process.cpu_times().user,
            "system": process.cpu_times().system,
        },
    }

    # Add platform-specific information
    try:
        info["num_fds"] = process.num_fds()
    except AttributeError:
        info["num_handles"] = (
            process.num_handles() if hasattr(process, "num_handles") else None
        )

    try:
        info["connections"] = len(process.connections())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        info["connections"] = None

    return info
