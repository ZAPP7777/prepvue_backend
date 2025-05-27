import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:  # Prevent adding multiple handlers
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
