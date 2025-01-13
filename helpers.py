import logging
import uuid
from datetime import datetime
import os


"""
Create logger

Parameters:
- module_name (str): Name of log file
- log_dir (path): Path to the log file

Returns:
- logger
"""
def setup_logger(module_name, log_dir="./logs"):
    """Set up a logger for a specific module."""
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    log_file = os.path.join(log_dir, f"{module_name}.log")    
    # Set the logging level (can be DEBUG, WARNING, ERROR, etc.)
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)  
    # Add handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add Handlers to Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


"""
Create subfolders for each module

Parameters:
- config (yaml): The configuration file.

Returns:
- Path to where all sessions are.
"""
def create_session_folder(config):
    base_path = config['Global']['session_base_path']
    """Create a unique session folder inside the base path."""
    # Generate a unique session ID using timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"{timestamp}-{uuid.uuid4().hex[:2]}"
    session_path = os.path.join(base_path, session_id)
    os.makedirs(session_path, exist_ok=True)
    run_modules = config['Modules']
    # Make folder for each modules
    for module_name, should_run in run_modules.items():
        if should_run:
            module_folder = os.path.join(session_path, module_name)
            os.makedirs(module_folder, exist_ok=True)
    return session_path


# """
# Validate the modules specified in the configuration.

# Parameters:
# - run_modules (dict): The 'Modules' section of the configuration.

# Raises:
# - ValueError: If invalid modules are found or no valid module is enabled.
# """
# def validate_modules(run_modules):
#     # Define allowed modules
#     ALLOWED_MODULES = ['Characterization', 'Harmonization', 'Robustness']
#     # Find invalid modules
#     invalid_modules = [module for module in run_modules if module not in ALLOWED_MODULES]
#     if invalid_modules:
#         invalid_str = ', '.join(invalid_modules)
#         allowed_str = ', '.join(ALLOWED_MODULES)
#         raise ValueError(
#             f"Invalid module(s) found: {invalid_str}. Allowed modules are: {allowed_str}"
#         )

#     # Check if at least one valid module is enabled
#     if not any(run_modules.get(module, False) for module in ALLOWED_MODULES):
#         allowed_str = ', '.join(ALLOWED_MODULES)
#         raise ValueError(
#             f"No valid module is enabled in 'run_modules'! At least one of the following must be set to true: {allowed_str}"
#         )
