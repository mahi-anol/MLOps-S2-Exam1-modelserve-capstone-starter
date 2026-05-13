import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class Logger:

    _loggers = {}

    @staticmethod
    def get_logger(name: str = __name__):
        """
        Get or create a named logger with console + file handlers.

        Args:
            name: identifier for the logger (typically __file__)
        Returns:
            logging.Logger instance
        """
        processed_name = Path(name).stem

        if processed_name in Logger._loggers:
            return Logger._loggers[processed_name]

        logger = logging.getLogger(processed_name)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        # File handler
        file_path = LOG_DIR / f"{processed_name}.log"
        file_handler = RotatingFileHandler(
            file_path, maxBytes=500_000, backupCount=2
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console)
        logger.addHandler(file_handler)
        logger.propagate = False

        Logger._loggers[processed_name] = logger
        return logger
