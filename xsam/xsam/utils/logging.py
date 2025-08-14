import logging
import os
import sys
from logging import Logger, LogRecord
from typing import Optional, Union

from mmengine.logging import MMLogger as _MMLogger


class FilterMasterRank(logging.Filter):
    """Filter log messages from non-master processes in distributed training.
    Args:
        name (str): name of the filter. Defaults to ''.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name)

    def filter(self, record: LogRecord) -> bool:
        """Filter the log message of non-master processes.
        Args:
            record (LogRecord): The log record.
        Returns:
            bool: True if the log is from master process (rank 0).
        """
        return int(os.environ.get("LOCAL_RANK", 0)) == 0


class XSamLogger(_MMLogger):
    def __init__(self, name="XSamLogger", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        for handler in self.handlers:
            handler.addFilter(FilterMasterRank())


def set_default_logging_format():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        formatter = logging.Formatter(
            fmt="[%(asctime)s,%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Set the format of root handlers
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout)

        logging.getLogger().handlers[0].addFilter(FilterMasterRank())
        logging.getLogger().handlers[0].setFormatter(formatter)
    else:
        logging.getLogger().addHandler(logging.NullHandler())
        logging.disable(logging.CRITICAL)


def print_log(msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = XSamLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if XSamLogger.check_instance_created(logger):
            logger_instance = XSamLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"XSamLogger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )
