import yaml
import sys
import argparse
from .helpers import *
import importlib
import time


# Load the config file
def load_config(config_path):
    """Load the configuration file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def main():
    """Main function to execute ctnorm with CLI arguments."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run Toolkit Modules")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    # Load the configuration file
    config = load_config(args.config)

    # Set up session folders
    session_path = create_session_folder(config)
    global_logger = setup_logger("base", session_path)
    global_logger.info(f"Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    global_logger.info(f"Session output will be stored in: {session_path}")
    global_logger.info("=" * 40)
    session_start_time = time.time()
    
    # Run module
    for module_name, should_run in config['Modules'].items():
        if should_run:
            global_logger.info(f"Running module: {module_name}")
            global_logger.info("-" * 40)
            start_time = time.time()
            module_path = f"ctnorm.{module_name}.run_module"
            module = importlib.import_module(module_path)
            module.main(config, global_logger, session_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            global_logger.info(f"Completed module {module_name} in {elapsed_time:.2f} seconds")

    session_end_time = time.time()
    total_elapsed_time = session_end_time - session_start_time
    global_logger.info("=" * 40)
    global_logger.info(f"Session completed successfully at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    global_logger.info(f"Total session duration: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
