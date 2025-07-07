import time
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, Response, request
from charset_normalizer import from_path

from lpm_kernel.api.domains.trainprocess.trainprocess_service import TrainProcessService
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from ...common.responses import APIResponse
from threading import Thread

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

trainprocess_bp = Blueprint("trainprocess", __name__, url_prefix="/api/trainprocess")

@trainprocess_bp.route("/start", methods=["POST"])
def start_process():
    """
    Start training process, returns progress stream ID
    
    Request parameters:
        model_name: Model name
        learning_rate: Learning rate for model training (optional)
        number_of_epochs: Number of training epochs (optional)
        concurrency_threads: Number of threads for concurrent processing (optional)
        data_synthesis_mode: Mode for data synthesis (optional)
        use_cuda: Whether to use CUDA for training (optional)
    
    Includes the following steps:
    1. Health check
    2. Generate L0
    3. Generate document embeddings
    4. Process document chunks
    5. Generate chunk embeddings
    6. Analyze documents
    7. Generate L1
    8. Download model
    9. Prepare data
    10. Train model
    11. Merge weights
    12. Convert model

    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "progress_id": "Progress stream ID",
                "model_name": "Model name"
            }
        }
    """
    logger.info("Training process starting...")  # Log the startup
    try:
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameters"))

        model_name = data["model_name"]
        
        # Get optional parameters with default values
        learning_rate = data.get("learning_rate", None)
        number_of_epochs = data.get("number_of_epochs", None)
        concurrency_threads = data.get("concurrency_threads", None)
        data_synthesis_mode = data.get("data_synthesis_mode", None)
        use_cuda = data.get("use_cuda", False)  # Default to False if not provided
        is_cot = data.get("is_cot", None)
        
        # Log the received parameters
        logger.info(f"Training parameters: model_name={model_name}, learning_rate={learning_rate}, number_of_epochs={number_of_epochs}, concurrency_threads={concurrency_threads}, data_synthesis_mode={data_synthesis_mode}, is_cot={is_cot}")

        # Create service instance with model name and additional parameters
        last_train_service = TrainProcessService.get_instance()
        
        # Check if there are any in_progress statuses that need to be reset
        if last_train_service is not None and last_train_service.progress.progress.data["status"] == "in_progress":
            return jsonify(APIResponse.error(
                message="There is an existing training process that was interrupted.",
                code=409  # Conflict status code
            ))
            

        train_service = TrainProcessService(current_model_name=model_name)
        if not train_service.check_training_condition():
            train_service.reset_progress()

        # Save training parameters
        training_params = {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "number_of_epochs": number_of_epochs,
            "concurrency_threads": concurrency_threads,
            "data_synthesis_mode": data_synthesis_mode,
            "use_cuda": use_cuda,  # Make sure to include use_cuda parameter
            "is_cot": is_cot
        }
        
        params_manager = TrainingParamsManager()
        # Update the latest training parameters
        params_manager.update_training_params(training_params)
        
        # Log training parameters
        logger.info(f"Saved training parameters: {training_params}")

        thread = Thread(target=train_service.start_process)
        thread.daemon = True
        thread.start()

        # Return success response with all parameters
        return jsonify(
            APIResponse.success(
                data={
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "number_of_epochs": number_of_epochs,
                    "concurrency_threads": concurrency_threads,
                    "data_synthesis_mode": data_synthesis_mode,
                    "use_cuda": use_cuda,  # Include in response
                    "is_cot": is_cot
                }
            )
        )
    
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        return jsonify(APIResponse.error(message=f"Training process error: {str(e)}"))

@trainprocess_bp.route("/logs", methods=["GET"])
def stream_logs():
    """Get training logs in real-time"""
    log_file_path = "logs/train/train.log"  # Log file path
    last_position = 0
    def generate_logs():
        nonlocal last_position
        while True:
            try:
                encoding = from_path(log_file_path).best().encoding
                with open(log_file_path, 'r', encoding=encoding) as log_file:
                    log_file.seek(last_position)
                    new_lines = log_file.readlines()  # Read new lines

                    for line in new_lines:
                        # Skip empty lines
                        if not line.strip():
                            continue
                        
                        yield f"data: {line.strip()}\n\n"
                            
                    last_position = log_file.tell()
                    if not new_lines:
                        yield f":heartbeat\n\n"
            except Exception as e:
                # If file reading fails, record error and continue
                yield f"data: Error reading log file: {str(e)}\n\n"
                
            time.sleep(1)  # Check for new logs every second

    return Response(
        generate_logs(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
    )

@trainprocess_bp.route("/progress/<model_name>", methods=["GET"])
def get_progress(model_name):
    """Get current progress (non-real-time)"""
    sanitized_model_name = secure_filename(model_name)  # Sanitize model_name
    try:
        train_service = TrainProcessService(current_model_name=sanitized_model_name)  # Pass in specific progress file
        progress = train_service.progress.progress

        return jsonify(
            APIResponse.success(
                data=progress.to_dict()  # Return progress data
            )
        )
    except Exception as e:
        logger.error(f"Get progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=str(e)))

@trainprocess_bp.route("/progress/reset", methods=["POST"])
def reset_progress():
    """
    Reset progress

    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": null
        }
    """
    try:
        train_service = TrainProcessService.get_instance()
        if train_service is not None:
            train_service.progress.reset_progress()
            logger.info("Progress reset successfully")
        else:
            logger.warning("No active training process found")

        return jsonify(APIResponse.success(message="Progress reset successfully"))
    except Exception as e:
        logger.error(f"Reset progress failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to reset progress: {str(e)}"))


@trainprocess_bp.route("/stop", methods=["POST"])
def stop_training():
    """Stop training process and wait until status is suspended"""
    try:
        # Get the TrainProcessService instance
        train_service = TrainProcessService.get_instance()  # Need to get instance based on your implementation
        if train_service is None:
            return jsonify(APIResponse.error(message="Failed to stop training: No active training process"))
        
        # Stop the process
        train_service.stop_process()
        
        # Wait for the status to change to SUSPENDED
        wait_interval = 1  # Check interval in seconds
        
        while True:
            # Get the current progress
            progress = train_service.progress.progress
            
            # Check if status is SUSPENDED
            if progress.data["status"] == "suspended" or progress.data["status"] == "failed":
                return jsonify(APIResponse.success(
                    message="Training process has been stopped and status is confirmed as suspended",
                    data={"status": "suspended"}
                ))
            
            # Wait before checking again
            time.sleep(wait_interval)

    except Exception as e:
        logger.error(f"Error stopping training process: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error stopping training process: {str(e)}"))

@trainprocess_bp.route("/step_output_content", methods=["GET"])
def get_step_output_content():
    """
    Get content of output file for a specific training step
    
    Request parameters:
        step_name: Name of the step to get content for, e.g. 'extract_dimensional_topics'
    
    Returns:
        Response: JSON response
        {
            "code": 0,
            "message": "Success",
            "data": {...}  // Content of the output file, or null if not found
        }
    """
    try:
        # Get TrainProcessService instance
        train_service = TrainProcessService.get_instance()
        if train_service is None:
            logger.error("No active training process found.")
            return jsonify(APIResponse.error(message="No active training process found."))
        
        # Get step name from query parameters
        step_name = request.args.get('step_name')
        if not step_name:
            return jsonify(APIResponse.error(message="Missing required parameter: step_name", code=400))
        
        # Get step output content
        output_content = train_service.get_step_output_content(step_name)  
        logger.info(f"Step output content: {output_content}")      
        return jsonify(APIResponse.success(data=output_content))
    except Exception as e:
        logger.error(f"Failed to get step output content: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to get step output content: {str(e)}"))

@trainprocess_bp.route("/training_params", methods=["GET"])
def get_training_params():
    """
    Get the latest training parameters
    
    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "model_name": "Model name",
                "learning_rate": "Learning rate",
                "number_of_epochs": "Number of epochs",
                "concurrency_threads": "Concurrency threads",
                "data_synthesis_mode": "Data synthesis mode"
            }
        }
    """
    try:
        # Get the latest training parameters
        params_manager = TrainingParamsManager()
        training_params = params_manager.get_latest_training_params()
        
        return jsonify(APIResponse.success(data=training_params))
    except Exception as e:
        logger.error(f"Error getting training parameters: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error getting training parameters: {str(e)}"))


@trainprocess_bp.route("/retrain", methods=["POST"])
def retrain():
    """
    Reset progress to data processing stage (data_processing not started) and automatically start the training process
    
    Request parameters:
        model_name: Model name (required)
        learning_rate: Learning rate for model training (optional)
        number_of_epochs: Number of training epochs (optional)
        concurrency_threads: Number of threads for concurrent processing (optional)
        data_synthesis_mode: Mode for data synthesis (optional)
        use_cuda: Whether to use CUDA for training (optional)
        is_cot: Whether to use Chain of Thought (optional)
    
    Returns:
        Response: JSON response
        {
            "code": 0 for success, non-zero for failure,
            "message": "Error message",
            "data": {
                "progress_id": "Progress stream ID",
                "model_name": "Model name"
            }
        }
    """
    try:
        # get request parameters
        data = request.get_json() or {}
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify(APIResponse.error(message="missing necessary parameter: model_name", code=400))
        
        # Get optional parameters
        learning_rate = data.get("learning_rate", None)
        number_of_epochs = data.get("number_of_epochs", None)
        concurrency_threads = data.get("concurrency_threads", None)
        data_synthesis_mode = data.get("data_synthesis_mode", None)
        use_cuda = data.get("use_cuda", False)
        is_cot = data.get("is_cot", None)
        
        # Log the received parameters
        logger.info(f"Retrain parameters: model_name={model_name}, learning_rate={learning_rate}, number_of_epochs={number_of_epochs}, concurrency_threads={concurrency_threads}, data_synthesis_mode={data_synthesis_mode}, use_cuda={use_cuda}, is_cot={is_cot}")
        
        # Create training service instance
        train_service = TrainProcessService(current_model_name=model_name)
        
        # Check if there are any in_progress statuses that need to be reset
        if train_service.progress.progress.data["status"] == "in_progress":
            # Reset the progress and continue
            logger.info("There is an existing training process that was interrupted.")
            
        train_service.reset_progress()

        # Save training parameters
        training_params = {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "number_of_epochs": number_of_epochs,
            "concurrency_threads": concurrency_threads,
            "data_synthesis_mode": data_synthesis_mode,
            "use_cuda": use_cuda,
            "is_cot": is_cot
        }
        
        params_manager = TrainingParamsManager()
        # Update the training parameters, optionally using previous params as base
        params_manager.update_training_params(training_params, use_previous_params=False)
        
        # Log training parameters
        logger.info(f"Saved training parameters: {training_params}")

        thread = Thread(target=train_service.start_process)
        thread.daemon = True
        thread.start()
        
        return jsonify(
            APIResponse.success(
                message="Successfully reset progress to data processing stage and started training process",
                data={
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "number_of_epochs": number_of_epochs,
                    "concurrency_threads": concurrency_threads,
                    "data_synthesis_mode": data_synthesis_mode,
                    "use_cuda": use_cuda,
                    "is_cot": is_cot
                }
            )
        )
    except Exception as e:
        logger.error(f"Retrain reset failed: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to reset progress to data processing stage: {str(e)}"))
