# Author: KleaSCM
# Date: 2024
# Description: Centralized logging utilities for Kasmeer system

import logging
import os
from functools import wraps
from time import time
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, f'kasmeer_{datetime.now().strftime("%Y%m%d")}.log')

# Setup the root logger

def setup_logging(name=None, level=logging.INFO):
    # Setup and return a logger with file and console handlers
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(ch)
    
    return logger

# Decorator to log performance and timing

def log_performance(logger=None, log_args=True, log_result=True):
    # Decorator to log function execution time, args, and result summary
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            start = time()
            log.info(f"CALL {func.__name__}()" + (f" args={args}, kwargs={kwargs}" if log_args else ""))
            try:
                result = func(*args, **kwargs)
                elapsed = time() - start
                if log_result:
                    log.info(f"RESULT {func.__name__}() => {str(result)[:200]}")
                log.info(f"PERF {func.__name__}() took {elapsed:.4f}s")
                return result
            except Exception as e:
                log.exception(f"ERROR in {func.__name__}(): {e}")
                raise
        return wrapper
    return decorator 