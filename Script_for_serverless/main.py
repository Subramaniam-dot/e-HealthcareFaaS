from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read())).convert('L')  # convert image to grayscale

    # Preprocess the image
    image = image.resize((224, 224))  # replace with the input size expected by your model
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)  # add an extra dimension for the color channel

    # Normalize the image data to the range [0, 1]
    image /= 255.

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Retrieve the output of the inference
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the class
    prediction = np.argmax(output_data, axis=1)

    # Map the prediction to the corresponding class
    class_names = ['Manifestation_of_Tuberculosis' ,'Normal_Lung'] # replace with your class names
    prediction_class = class_names[prediction[0]]

    return {"prediction": prediction_class}
