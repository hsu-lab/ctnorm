import yaml
import sys
import argparse
from helpers import *
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


if __name__ == "__main__":
    # Parse CLI
    parser = argparse.ArgumentParser(description="Run Toolkit Modules")
    parser.add_argument('--opt', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.opt)

    # Set up session folders
    session_path = create_session_folder(config)
    global_logger = setup_logger("base", session_path)
    global_logger.info(f"Session output at: {session_path}")
    
    # Run module
    for module_name, should_run in config['Modules'].items():
        if should_run:
            global_logger.info(f"Running module: {module_name}")
            # try:
            start_time = time.time()
            module_path = f"{module_name}.run_module"
            module = importlib.import_module(module_path)
            module.main(config, global_logger, session_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            global_logger.info(f"Completed module {module_name} in {elapsed_time:.2f} seconds")
            # except ModuleNotFoundError:
            #     global_logger.error(f"Module {module_name} not found!")
            # except Exception as e:
            #     global_logger.error(f"Error while running module {module_name}: {e}")