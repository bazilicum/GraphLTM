"""
Utilities for logging.
"""
import logging
import time
from typing import Optional, List, Dict
from config import Config

#load config and setup logging
config = Config()

def setup_logging(config: Config, module_name: str) -> logging.Logger:
    """
    Set up logging with a custom formatter for aligned output.
    
    Args:
        config: Configuration object
        module_name: Name of the module (typically __name__)
        
    Returns:
        Logger with custom formatting
        
    Raises:
        ValueError: If invalid log level is specified
    """
    # Get the logger for the specific module
    logger = logging.getLogger(module_name)
    
    # Only add handler if not already added to prevent duplicate logs
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        
        # Custom formatter with alignment
        class AlignedFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                """
                Format the log record with aligned output.
                
                Args:
                    record: The log record to format
                    
                Returns:
                    Formatted log message string
                """
                # Get current timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))

                # Extract module name, truncate if too long
                module = record.name.split('.')[-1]
                module = (module[:15] + '..') if len(module) > 17 else module
                
                # Pad module to a fixed width
                module = module.ljust(17)
                
                # Pad log level to a fixed width
                level = record.levelname.ljust(7)
                
                # Create the formatted message
                return f"{timestamp} | {module} | {level} | {record.getMessage()}"
        
        # Apply the custom formatter
        handler.setFormatter(AlignedFormatter())
        
        # Set log level based on configuration
        log_level = config.get('logging', 'level').upper()
        if not hasattr(logging, log_level):
            raise ValueError(f"Invalid log level: {log_level}")
        logger.setLevel(getattr(logging, log_level))
        
        # Add the handler
        logger.addHandler(handler)
    
    return logger

def print_framed(text: str, border_char: str = "=", border_length: int = 80) -> None:
    """
    Prints the given text framed by a border for clear visual output.
    
    Args:
        text: Text to print
        border_char: Character to use for the border (default: "=")
        border_length: Length of the border (default: 80)
    """
    border = border_char * border_length
    print(border)
    print(text)
    print(border)

def print_message_list(message_list: List[Dict[str, str]]) -> None:
    """
    Print a list of messages in a formatted way.
    
    Args:
        message_list: List of message dictionaries with 'role' and 'content' keys
    """
    if not message_list:
        print("No messages to display")
        return
        
    for message in message_list:
        if 'role' not in message or 'content' not in message:
            print("Invalid message format: missing role or content")
            continue
            
        print("-" * 20)
        print(f"{message['role']}: {message['content']}")
        print("-" * 20)