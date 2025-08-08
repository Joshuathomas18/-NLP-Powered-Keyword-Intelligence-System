"""Logging utilities."""

import logging
import json
from datetime import datetime
from pathlib import Path


def setup_logging(level: str = "INFO", output_dir: str = "./outputs") -> str:
    """Setup structured logging with file and console handlers."""
    
    # Create run ID
    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    
    # Setup logging directory
    log_dir = Path(output_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler with JSON format
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return run_id


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
            
        if hasattr(record, 'url'):
            log_entry['url'] = record.url
            
        if hasattr(record, 'keywords_count'):
            log_entry['keywords_count'] = record.keywords_count
            
        return json.dumps(log_entry)


def get_logger(component: str) -> logging.Logger:
    """Get logger with component context."""
    logger = logging.getLogger(component)
    
    # Add component context to all log records
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.component = component
        return record
    
    logging.setLogRecordFactory(record_factory)
    return logger