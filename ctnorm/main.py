import yaml
import sys
import argparse
import json
import importlib
import time
from .helpers import *


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


def update_session_status(session_path, status, config_file=None, module_status=None, error_messages=None):
    """Update the session status and module progress in a JSON file."""
    status_file = os.path.join(session_path, "session_status.json")
    # Load existing status file if it exists
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status_data = json.load(f)
    else:
        status_data = {}
    # Update fields
    status_data.update({
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_path.split('/')[-1],
        "config_file": config_file if config_file else status_data.get("config_file"),
        "module_status": module_status if module_status else status_data.get("module_status", {}),
        "error_messages": error_messages if error_messages else status_data.get("error_messages", {}),  # ✅ Store error messages per module
        "elapsed_time": round(time.time() - status_data.get("start_time", time.time()), 2),
    })

    # If the session is starting, store the start time
    if status == "running":
        status_data["start_time"] = time.time()
    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=4)



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

    # Initialize module tracking dictionary
    module_status = {module_name: "pending" for module_name, should_run in config['Modules'].items() if should_run}
    error_messages = {}  # Dictionary to store error messages for each failed module

    # Mark session as running
    update_session_status(session_path, "running", config_file=args.config, module_status=module_status, error_messages=error_messages)
    session_start_time = time.time()

    for module_name in module_status.keys():
        global_logger.info(f"Running module: {module_name}")
        global_logger.info("-" * 40)

        start_time = time.time()
        module_path = f"ctnorm.{module_name}.run_module"
        module = importlib.import_module(module_path)

        try:
            module.main(config, global_logger, session_path)
            # Mark module as completed
            module_status[module_name] = "completed"
        except Exception as module_error:
            global_logger.error(f"Module {module_name} encountered an error: {module_error}")
            module_status[module_name] = "failed"
            error_messages[module_name] = str(module_error)  # Store error message for this module

        # Update session status after each module
        update_session_status(session_path, "running", module_status=module_status, error_messages=error_messages)

        end_time = time.time()
        elapsed_time = end_time - start_time
        global_logger.info(f"Finished module {module_name} in {elapsed_time:.2f} seconds")

    # Determine overall session status
    session_status = "failed" if any(status == "failed" for status in module_status.values()) else "completed"
    # Update final session status
    update_session_status(session_path, session_status, module_status=module_status, error_messages=error_messages)
    session_end_time = time.time()
    total_elapsed_time = session_end_time - session_start_time
    global_logger.info("=" * 40)
    global_logger.info(f"Session {session_status} at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    global_logger.info(f"Total session duration: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
