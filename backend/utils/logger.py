"""Production logging configuration"""
import logging
import sys
from pathlib import Path
from loguru import logger
from backend.config import settings

def setup_logging():
    """Configure production logging"""
    
    # Remove default handler
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO" if not settings.DEBUG else "DEBUG"
    )
    
    # File logging
    log_file = settings.LOGS_DIR / "sovereign_ai_{time:YYYY-MM-DD}.log"
    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    # Error logging
    error_log = settings.LOGS_DIR / "errors_{time:YYYY-MM-DD}.log"
    logger.add(
        error_log,
        rotation="1 day",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}\n{exception}"
    )
    
    return logger

# Initialize logger
log = setup_logging()