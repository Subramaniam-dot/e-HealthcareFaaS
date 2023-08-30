
# Binary Classification: Let's do this!!!

# Transfer Learning

# Go to Working Directory
"""



"""# Make batches of the Data"""

# Assign train and test directories

train_dir = "Dataset/train"
test_dir = 'Dataset/test'

# Create data inputs
import tensorflow as tf

IMG_SIZE = (224, 224) # define image size
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="binary", # what type are the labels?
                                                                            color_mode = "grayscale",
                                                                            batch_size=32) # batch_size is 32 by default, this is generally a good number
test_data= tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                           image_size=IMG_SIZE,
                                                                           label_mode="binary",
                                                                            color_mode = "grayscale",
                                                               batch_size=32)

"""```
==> Data Augmentation is usually only performed on the training Data.

Using Image DataGenerator built in data augmentation parameters our
images are left as they are in the directories but are modified as
they're loaded into the model.



```
"""

#Create TensorBoard CallBacks (Functionized because we need to create a new one for each model)

import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
  print(f"Saving TensorBoard Log files to: {log_dir}")
  return tensorboard_callback

#Import dependencies

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

class ExpandChannelLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpandChannelLayer, self).__init__(**kwargs)

    def call(self, input):
        return tf.tile(input, [1, 1, 1, 3])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 3)

import tensorflow as tf

input_shape = (224,224,1)

base_model = tf.keras.applications.MobileNetV3Small(include_top=False)
base_model.trainable =False

inputs = tf.keras.layers.Input(shape = input_shape, name = "input_layer")

x = ExpandChannelLayer(name = "expand_channel_layer")(inputs)
x = base_model(x, training = False)
x = tf.keras.layers.GlobalAveragePooling2D(name = "globalAveragePooling2D")(x)
outputs = tf.keras.layers.Dense(1,activation ="sigmoid", name = "output_layer")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
              metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.BinaryIoU(), tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])

from helper_functions import create_tensorboard_callback, plot_loss_curves

"""#ModelCheckpoint"""

#set checkpoint path

checkpoint_path = "modelcheckpoint_weights_Mobilenet_feature+finetune/checkpoint.ckpt"

#create a ModelCheckpoint Callback that saves the model's weight only

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                   save_weights_only = True,
                                   save_best_only = True,
                                   save_freq = "epoch",
                                   verbose =1)

initial_epochs = 15
history_resnet = model.fit(train_data,
                        epochs = initial_epochs,
                        steps_per_epoch = len(train_data),
                        validation_data = test_data,
                        validation_steps = len(test_data),
                         callbacks = [create_tensorboard_callback(dir_name = "MobilenetV3",
                                                                  experiment_name = "MobilenetV3"),
                                      checkpoint_callback
                                      ])

model.evaluate(test_data)

import matplotlib.pyplot as plt

def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # Plot training & validation accuracy values
    axs[0].plot(history.history['binary_accuracy'])
    axs[0].plot(history.history['val_binary_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()

# Assuming that 'history' is the returned value from the 'fit' operation
plot_history(history_resnet)

# Get the values from the history object
TP = history_resnet.history['val_true_positives'][-1]
TN = history_resnet.history['val_true_negatives'][-1]
FP = history_resnet.history['val_false_positives'][-1]
FN = history_resnet.history['val_false_negatives'][-1]

# Calculate Precision, Recall, F1 Score
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * ((Precision * Recall) / (Precision + Recall))

print("Precision: ", Precision)
print("Recall: ", Recall)
print("F1 Score: ", F1_Score)

model.load_weights(checkpoint_path)

loaded_weight_model_results = model.evaluate(test_data)

#Are these Layers Trainable?
for layer in model.layers:
  print(layer, layer.trainable)

for i,layer in enumerate(model.layers[2].layers):
  print(i, layer.name, layer.trainable)

# Unfreeze the feature extraction layer
base_model.trainable = True

for layer in base_model.layers[:-3]:
  layer.trainable = False

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.BinaryIoU(), tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])

fine_tune_epochs = initial_epochs + 5
# Continue training
history_2 = model.fit(train_data,
                                 epochs= fine_tune_epochs,
                                         validation_data = test_data,
                                         validation_steps = len(test_data),
                                         steps_per_epoch = len(train_data),
                                         initial_epoch= history_resnet.epoch[-1],
                                         callbacks = [create_tensorboard_callback(dir_name = "MobilenetV3",
                                                                  experiment_name = "MobilenetV3"),
                                                      ]
                                           )

loaded_weight_model_results = model.evaluate(test_data)

#Lets create a function to compare training histories
import matplotlib.pyplot as plt
def compare_history(original_history, new_history, initial_epochs = 5):
  """
  Compare two TensorFlow history objects
  """

  #Get original history metrics
  acc = original_history.history["binary_accuracy"]
  loss = original_history.history["loss"]

  val_acc = original_history.history["val_binary_accuracy"]
  val_loss = original_history.history["val_loss"]

  #Combine origibal history metrics with new_history metrics

  total_acc = acc + new_history.history["binary_accuracy"]
  total_loss = loss + new_history.history["loss"]

  total_val_acc = val_acc + new_history.history["val_binary_accuracy"]
  total_val_loss= val_loss + new_history.history["val_loss"]

  # Make Plots
  plt.figure(figsize=(10,7))
  plt.subplot(2,1,1)
  plt.plot(total_acc, label ="Training Accuracy")
  plt.plot(total_val_acc, label = "VAl Accuracy")
  plt.plot([initial_epochs -1, initial_epochs -1], plt.ylim(), label = "Start Fine Tuning" )
  plt.legend(loc = "lower right")
  plt.title("training and validation Accuracy")

    # Make Plots
  plt.figure(figsize=(10,7))
  plt.subplot(2,1,1)
  plt.plot(total_loss, label ="Training loss")
  plt.plot(total_val_loss, label = "VAl loss")
  plt.plot([initial_epochs -1, initial_epochs -1], plt.ylim(), label = "Start Fine Tuning" )
  plt.legend(loc = "lower right")
  plt.title("training and validation loss")

# Get the values from the history object
TP = history_2.history['val_true_positives_1'][-1]
TN = history_2.history['val_true_negatives_1'][-1]
FP = history_2.history['val_false_positives_1'][-1]
FN = history_2.history['val_false_negatives_1'][-1]

# Calculate Precision, Recall, F1 Score
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * ((Precision * Recall) / (Precision + Recall))

print("Precision: ", Precision)
print("Recall: ", Recall)
print("F1 Score: ", F1_Score)

compare_history(history_resnet,history_2, initial_epochs = 15)

model.save('Mobilenetv3-Featureextract_Finetune-model1')

test_accuracy = history_2.history['val_binary_accuracy'][-1]
test_loss = history_2.history['val_loss'][-1]
test_io_u = history_2.history['val_binary_io_u_1'][-1]

with open('test_metrics_mobilenet.txt', 'w') as f:
    f.write(f'Test Accuracy: {test_accuracy}\n')
    f.write(f'Test Loss: {test_loss}\n')
    f.write(f'Test IoU: {test_io_u}\n')
    f.write(f'Test Precision: {Precision}\n')
    f.write(f'Test Recall: {Recall}\n')
    f.write(f'Test F1 Score: {F1_Score}\n')

# from google.colab import runtime
# runtime.unassign()

