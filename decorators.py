import logging
import time
from functools import wraps

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        st = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - st

        logger.info(
            f"""Func: {func.__name__} with args: {args} kwargs: {kwargs} executed in: {elapsed_time:.3f}  secs."""
        )
        return result

    return timeit_wrapper
