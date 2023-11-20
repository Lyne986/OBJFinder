#!/bin/bash

# Set the model name and directory
MODEL_NAME="centernet_hg104_512x512_coco17_tpu-8"
MODEL_DIR="data/model/$MODEL_NAME"

# Check if the model directory already exists
if [ -d "$MODEL_DIR" ]; then
    echo "Model directory '$MODEL_DIR' already exists. Skipping download."
else
    # Create the model directory if it doesn't exist
    mkdir -p $MODEL_DIR

    # Download the model from TensorFlow Model Zoo
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL_NAME.tar.gz -O $MODEL_NAME.tar.gz

    # Extract the downloaded model to a temporary directory
    TEMP_DIR=$(mktemp -d)
    tar -xzvf $MODEL_NAME.tar.gz -C $TEMP_DIR

    # Move the contents up one level
    mv $TEMP_DIR/$MODEL_NAME/* $MODEL_DIR

    # Remove the temporary directory and the downloaded tar.gz file
    rm -r $TEMP_DIR
    rm $MODEL_NAME.tar.gz

    echo "Model downloaded and extracted to $MODEL_DIR"
fi