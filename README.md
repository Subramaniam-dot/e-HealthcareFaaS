<h1 align="center">e-HealthcareFaaS</h1>
<div align="center">
  <a href="https://github.com/your-repo/e-HealthcareFaaS/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
  <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyour-repo%2Fe-HealthcareFaaS&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
  <a href="https://github.com/your-repo/e-HealthcareFaaS/actions">
    <img src="https://github.com/your-repo/e-HealthcareFaaS/workflows/CI/badge.svg" alt="Actions Status">
  </a>
  <a href="https://doi.org/10.1109/JBHI.2024.3367736">
    <img src="https://img.shields.io/badge/DOI-10.1109%2FJBHI.2024.3367736-green.svg" alt="DOI">
  </a>
</div>

> Neural Networks based Smart e-Health Application for the Prediction of Tuberculosis using Serverless Computing

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

_Note_: Similar steps apply for `vgg19.py` and `mobilenetv3small.py`.

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

   ```

   docker build -t subrome1305/api-lite:latest .

   ```

2. **Push Docker Image to Registry** (assuming DockerHub):

   ```
   docker push subrome1305/api-lite:latest
   ```

3. **Apply Kubernetes Configuration**:
   Make sure you have `kubectl` set up and configured to interact with your cluster. Then, apply the configurations:

   ```

   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

4. **Access the FastAPI Service**:
   If you're running on a cloud provider that supports load balancers, the `LoadBalancer` service type will provision an external IP for accessing the service. To get this IP:

   ```
   kubectl get svc my-service -n my-namespace
   ```

   Look for the `EXTERNAL-IP` column to get the IP. Once you have it, you can make requests to `http://<EXTERNAL-IP>/predict` to interact with your FastAPI application.

## Note:

- Ensure that your Kubernetes cluster has enough resources to handle the number of replicas specified in `deployment.yaml`.
- Update the `class_names` in `main.py` if your TFLite model class labels differ.
- Ensure you have the TFLite model (`model_quant.tflite`) in the same directory or adjust the path accordingly in `main.py`.

## Contributors:

1. [Subramaniam-dot](https://github.com/Subramaniam-dot)

2. [sasidharan01](https://github.com/thesouldev)
