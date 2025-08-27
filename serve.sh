#!/bin/bash
# src/serve.sh
# This script starts the TensorFlow Serving Docker container to serve our model.

# --- Configuration ---
# The absolute path to the directory containing the model versions.
# 'pwd' gets the current working directory, which should be the project root.
MODEL_BASE_PATH="$(pwd)/models"
MODEL_NAME="mnist"
# The port on the host machine that will be mapped to the container's REST API port.
HOST_PORT=8501

# --- Pre-flight Check ---
# Check if the models directory exists.
if [ ! -d "$MODEL_BASE_PATH/$MODEL_NAME" ]; then
    echo "ERROR: Model directory not found at '$MODEL_BASE_PATH/$MODEL_NAME'"
    echo "Please run 'python src/train.py' first to export the model."
    exit 1
fi

echo "--- Starting TensorFlow Serving ---"
echo "Model Name:       $MODEL_NAME"
echo "Model Base Path:  $MODEL_BASE_PATH"
echo "Host Port:        $HOST_PORT"
echo "REST API URL:     http://localhost:$HOST_PORT/v1/models/$MODEL_NAME:predict"
echo "------------------------------------"

# --- Run Docker Container ---
# -t: Allocate a pseudo-TTY
# --rm: Automatically remove the container when it exits
# -p: Publish a container's port(s) to the host. Format is hostPort:containerPort
# -v: Bind mount a volume. This maps our local models directory to the container's expected models directory.
# -e MODEL_NAME: An environment variable that TF Serving uses to find the model to serve.
# tensorflow/serving: The official Docker image.
docker run -t --rm -p ${HOST_PORT}:8501 \
    -v "${MODEL_BASE_PATH}/${MODEL_NAME}:/models/${MODEL_NAME}" \
    -e MODEL_NAME=${MODEL_NAME} \
    tensorflow/serving

# To stop the server, press Ctrl+C in this terminal.
