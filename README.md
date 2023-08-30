
# TB Diagnosis Model and Deployment Directory

This repository contains code for binary classification models for images, specifically using Transfer Learning. 

For Dataset Refer: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

## Directory Structure:

```
.
├── README.md
├── model
│   ├── densenet201.py
│   ├── vgg19.py
│   └── mobilenetv3small.py
├── model_to_tflite_conversion
│   └── tfmodel_to_tflite.py
├── script_for_ec2_server
│   ├── main.py
│   └── requirements.txt
└── Script_for_serverless
    ├── deployment.yaml
    ├── Dockerfile
    ├── main.py
    ├── requirements.txt
    └── service.yaml
```
## Instructions for Models - Densenet201, VGG19, MobilenetV3-Small:

1. Go to the working directory.
2. The data should be organized into `Dataset/train` for training images and `Dataset/test` for test images.
3. Data augmentation is only performed on the training data.
4. Run the `densenet201.py` script for training, evaluating, and plotting the model's performance.
5. The model's weights will be saved in the `modelcheckpoint_weights_Densenet_feature+finetune/` directory.

*Note*: Similar steps apply for `vgg19.py` and `mobilenetv3small.py`.

## Model Conversion to TensorFlow Lite:

1. Load the required model using the `load_model` function from `tensorflow.keras.models`.
2. To test the model, use a sample image named `test.png`.
3. Convert the model to TensorFlow Lite format.
4. For quantized conversion, use the optimization provided in the script.
5. The converted models will be saved as `model.tflite` and `model_quant.tflite`.

## EC2 Server Script:

1. Ensure all the required Python packages listed in `requirements.txt` are installed.
2. Run the `main.py` script to start the FastAPI server.
3. Use the provided endpoint to upload files and get predictions.


# Deployment Guide for FastAPI Application with Serverless Backend

This guide will help you deploy a FastAPI application, which performs image classification based on a TensorFlow Lite model. The deployment will be handled in a Kubernetes cluster using the provided deployment configurations.


### Files Explanation:

- **deployment.yaml**: This is a Kubernetes deployment configuration file. It defines the desired state for our application deployment, including the Docker image to be used, the number of pod replicas, and more.

- **Dockerfile**: Instructions for Docker to build our FastAPI application image. It specifies everything that is needed, from the base image to use, dependencies to install, and how to run our application.

- **main.py**: This contains the main FastAPI application logic. It includes routes, TFLite model loading, and inference.

- **requirements.txt**: A list of Python packages required for our FastAPI application.

- **service.yaml**: Kubernetes service configuration. It exposes our deployment to the network, either within the cluster or externally.

## Steps to Deploy:

1. **Build the Docker Image**:
   Navigate to the `Script_for_serverless` directory and run:
   \
   ```
   bash
   docker build -t subrome1305/api-lite:latest .
   \
   ```

3. **Push Docker Image to Registry** (assuming DockerHub):
   \```bash
   docker push subrome1305/api-lite:latest
   \```

4. **Apply Kubernetes Configuration**:
   Make sure you have `kubectl` set up and configured to interact with your cluster. Then, apply the configurations:
   \```
   bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   \```

5. **Access the FastAPI Service**:
   If you're running on a cloud provider that supports load balancers, the `LoadBalancer` service type will provision an external IP for accessing the service. To get this IP:
   \```bash
   kubectl get svc my-service -n my-namespace
   \```
   Look for the `EXTERNAL-IP` column to get the IP. Once you have it, you can make requests to `http://<EXTERNAL-IP>/predict` to interact with your FastAPI application.

## Note:
- Ensure that your Kubernetes cluster has enough resources to handle the number of replicas specified in `deployment.yaml`.
- Update the `class_names` in `main.py` if your TFLite model class labels differ.
- Ensure you have the TFLite model (`model_quant.tflite`) in the same directory or adjust the path accordingly in `main.py`.

