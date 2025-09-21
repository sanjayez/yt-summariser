"""Status calculation and result processing utilities"""


def process_batch_results(
    results: list[int | BaseException], micro_batch_count: int
) -> dict[str, int]:
    """Process parallel batch results and calculate success/failure metrics"""
    # Split results by batch type
    micro_results = results[:micro_batch_count]
    macro_results = results[micro_batch_count:]

    # Calculate success counts (sum integers)
    micro_upserted = sum(r for r in micro_results if isinstance(r, int))
    macro_upserted = sum(r for r in macro_results if isinstance(r, int))

    # Calculate failure counts (count exceptions)
    micro_failures = sum(1 for r in micro_results if isinstance(r, BaseException))
    macro_failures = sum(1 for r in macro_results if isinstance(r, BaseException))

    return {
        "micro_upserted": micro_upserted,
        "macro_upserted": macro_upserted,
        "micro_failures": micro_failures,
        "macro_failures": macro_failures,
    }


def calculate_status(upserted: int, total: int) -> str:
    """Calculate operation status based on success counts"""
    if upserted == 0 and total > 0:  # If total is 0, it's not a failure
        return "failed"
    elif upserted == total:
        return "success"
    elif upserted > 0 and upserted < total:
        return "partial"
    else:  # Case where total is 0, and upserted is 0
        return "success"  # Or "empty" depending on desired semantics
