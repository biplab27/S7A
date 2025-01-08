import logging
import os
import sys
from datetime import datetime

def setup_logger(log_dir='logs', force_new=False):
    """Set up logger that outputs to both file and console"""
    
    # Get existing logger if it exists
    logger = logging.getLogger('mnist_training')
    if logger.handlers and not force_new:  # If logger is already configured
        return logger
        
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log file name without timestamp
    log_file = os.path.join(log_dir, 'mnist_training.log')
    
    # Set up logging format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add program start marker only for new logger
    logger.info("="*80)
    logger.info("PROGRAM START")
    logger.info("="*80)
    
    return logger 