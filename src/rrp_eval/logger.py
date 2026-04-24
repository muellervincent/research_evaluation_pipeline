import sys
from loguru import logger
from pathlib import Path

def setup_logging(log_file: Path = None):
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week"
        )
    
    # Mute noisy google API logs
    import logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
            
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING, force=True)
    for noisy in ["google", "google.genai", "urllib3", "httpx"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

def print_cognitive_step(title: str, content: str):
    logger.info(f"🧠 [{title}]: {content}")

def highlight_print(message: str):
    logger.info(f"✨ {message}")
