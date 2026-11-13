"""Function execution timing."""

from functools import wraps
from time import perf_counter


def timelogger(func):
    """Print elapsed seconds for each call, including calls that raise."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        started = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            print(f"{func.__qualname__} took {elapsed:.6f} s")

    return wrapper
