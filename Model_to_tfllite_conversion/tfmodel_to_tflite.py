
from tensorflow.keras.models import load_model

model = load_model('VGG-Featureextract_Finetune-model1') #model name

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image



# Load an image file
img_path = 'test.png'
img = Image.open(img_path).convert('L')  # convert image to grayscale
img = img.resize((224, 224))  # replace with the input size expected by your model

# Convert the image to a numpy array
img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)
img_array = np.expand_dims(img_array, axis=-1)  # add an extra dimension for the color channel

# Normalize the image data to the range [0, 1]
img_array /= 255.


# Normalize the image data to the range [0, 1]
img_array /= 255.

# Make prediction
prediction = model.predict(img_array)
prediction = np.argmax(prediction, axis=1)  # get the class

    # Map the prediction to the corresponding class
class_names = ['Manifestation_of_Tuberculosis' ,'Normal_Lung'] # replace with your class names
prediction_class = class_names[prediction[0]]

print(f"prediction class:{prediction_class}")

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('model.tflite','wb') as f:
  f.write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized TF Lite model.
with tf.io.gfile.GFile('model_quant.tflite', 'wb') as f:
  f.write(tflite_quant_model)

import tensorflow as tf
import numpy as np

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on some input data.
input_shape = input_details[0]['shape']
input_data = img_array
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Compare the output_data with the expected output data (ground truth) to calculate accuracy
prediction = np.argmax(output_data, axis=1)  # get the class

    # Map the prediction to the corresponding class
class_names = ['Manifestation_of_Tuberculosis' ,'Normal_Lung'] # replace with your class names
prediction_class = class_names[prediction[0]]

print(f"prediction class:{prediction_class}")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/LungVision/VGG19-model/API_lite

# Commented out IPython magic to ensure Python compatibility.
# %ls

# !pip3 install -r requirements.txt



!uvicorn main:app --host 0.0.0.0 --port 8000

