import time
from typing import Any
from functools import partial

nesting_level = 0


def log(entry: Any, logger):
    """log entry

    Args:
        entry (Any): log content
        logger (logging.logger): logger object
    """
    global nesting_level
    space = "-" * (4 * nesting_level)
    if logger is None:
        print(f"{space}{entry}")
    else:
        logger.info(f"{space}{entry}")


def timeit(method=None, start_log=None, logger=None):
    """decorator for timing method

    Args:
        method (function): function name
        start_log (str): log mention
        logger (logging.logger): logger for log

    Returns:
        function: wrapped function
    """
    if method is None:
        return partial(timeit, start_log=start_log, logger=logger)

    def wrapper(*args, **kw):
        global nesting_level

        log(f"Start [{method.__name__}]: " + (start_log if start_log else ""), logger)
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.", logger)
        return result

    return wrapper
