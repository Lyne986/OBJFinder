#!/bin/bash

# Set the model names and directories
DETECTION_MODEL_NAME="centernet_hg104_512x512_coco17_tpu-8"
SEGMENTATION_MODEL_NAME="vgg16"
DETECTION_MODEL_DIR="data/model/$DETECTION_MODEL_NAME"
SEGMENTATION_MODEL_DIR="data/model/$SEGMENTATION_MODEL_NAME"

# Check if the detection model directory already exists
if [ -d "$DETECTION_MODEL_DIR" ]; then
    echo "Object detection model directory '$DETECTION_MODEL_DIR' already exists. Skipping download."
else
    # Create the detection model directory if it doesn't exist
    mkdir -p $DETECTION_MODEL_DIR

    # Download the object detection model from TensorFlow Model Zoo
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/$DETECTION_MODEL_NAME.tar.gz -O $DETECTION_MODEL_NAME.tar.gz

    # Extract the downloaded model to a temporary directory
    TEMP_DIR=$(mktemp -d)
    tar -xzvf $DETECTION_MODEL_NAME.tar.gz -C $TEMP_DIR

    # Move the contents up one level
    mv $TEMP_DIR/$DETECTION_MODEL_NAME/* $DETECTION_MODEL_DIR

    # Remove the temporary directory and the downloaded tar.gz file
    rm -r $TEMP_DIR
    rm $DETECTION_MODEL_NAME.tar.gz

    echo "Object detection model downloaded and extracted to $DETECTION_MODEL_DIR"
fi

# Check if the segmentation model directory already exists
if [ -d "$SEGMENTATION_MODEL_DIR" ]; then
    echo "Segmentation model directory '$SEGMENTATION_MODEL_DIR' already exists. Skipping download."
else
    # Create the segmentation model directory if it doesn't exist
    mkdir -p $SEGMENTATION_MODEL_DIR

    # Download the segmentation model (VGG16 in this case, adjust as needed)
    wget https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 -O $SEGMENTATION_MODEL_DIR/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

    echo "Segmentation model downloaded to $SEGMENTATION_MODEL_DIR"
fi
