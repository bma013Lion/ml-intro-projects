import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets

# Create output directory
os.makedirs('outputs/tensorflow', exist_ok=True)

# set random seed
tf.random.set_seed(42)

# check if GPU is running
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# hyper parameters
# batch size controls the number of training examples that are used to compute the gradient of the loss function in each iteration
batch_size = 64
# The learning rate is a hyperparameter that controls how quickly the model is adapted to the problem
learning_rate = 0.001
# The number of epochs is the number of times the model is trained on the entire dataset
num_epochs = 10

# load dataset FashionMNIST
# ---------------------------
# The dataset is divided into a training set and a test set
# The training set is used to train the model, while the test set is used to evaluate its performance
# The load_data() function returns the datasets as numpy arrays
# The first argument is a tuple of the training images and labels
# The second argument is a tuple of the test images and labels
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# add a channel dimension to the images
# ---------------------------
# The images are 2D arrays, with shape (height, width), so we need to add a channel dimension to them
# This is done to match the expected input shape of the model
# The channel dimension is added at the end of the array, and is of size 1
# This is why we use the tf.newaxis argument, which is equivalent to None
# This adds a new axis to the array, with size 1
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# image class labels for FashionMNIST
image_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# display some training images
plt.figure(figsize=(10, 5))
for i in range(14):
    plt.subplot(2, 7, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].squeeze(), cmap=plt.cm.binary)
    plt.title(image_labels[train_labels[i]])
plt.tight_layout()
plt.savefig("outputs/tensorflow/fashion_mnist_samples_tensorflow.png")
plt.close()

# define the neural network model
# ---------------------------
# This function creates the model
# It is called later in the code
# The model is a linear stack of layers
# The layers are added to the model in the order they are defined
# The model is compiled with a loss function and an optimizer
# The model is then returned
def create_model():
    """
    Create a neural network model for image classification.

    The model is a linear stack of layers, with two hidden layers.
    The first hidden layer has 512 neurons and an activation function of 
    'relu', which is the Rectified Linear Unit function. 
        - This is a simple non-linear activation function where all the negative 
        values are set to 0 and all the positive values are left as is.
    The second hidden layer has 256 neurons and an activation function of 'relu'.
    The output layer has 10 neurons, one for each class.
    There is no activation function on the output layer.
    """
    
    # define the model as a Sequential object
    # it will be a linear stack of layers
    model = models.Sequential()

    # add a Flatten layer to the model
    # this layer takes the 28x28x1 input and flattens it to 784
    model.add(layers.Flatten(input_shape=(28, 28, 1)))

    # add two Dense layers with 512 and 256 neurons respectively
    # the activation function is set to 'relu'
    # this is a common choice for the activation function for hidden layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))

    # add the final layer with 10 neurons for the 10 classes
    # the activation function is not set, so the output is a linear function
    # this is because the output should be a probability distribution
    # and the softmax function will be applied to the output in the loss function
    model.add(layers.Dense(10))

    # compile the model with the Adam optimizer
    # the learning rate is set to 0.01
    # the loss function is SparseCategoricalCrossentropy
    # this is a common choice for classification problems
    # the metrics is accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# create and train the model
model= create_model()

# print the model summary
print("\nModel Summary:")
model.summary()

# train the model
print("\nTraining the model...")
history = model.fit(
    train_images, train_labels,  # Use the NumPy arrays directly
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(test_images, test_labels)  # Also update validation data
)

# plot training history
plt.figure(figsize=(12, 4))

# plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/tensorflow/training_history.png")
plt.close()

# evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# plot some test images with their predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)    
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(image_labels[predicted_label],
                                        100*np.max(predictions_array),
                                        image_labels[true_label]),
                color=color)
    
# plot the first X test images, their predicted labels, and the true labels
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
plt.tight_layout()
plt.savefig("outputs/tensorflow/test_images_tensorflow.png")
plt.close()

# save the model
model.save("outputs/tensorflow/img_classifier_tensorflow.keras")
print("\nModel saved as img_classifier_tensorflow.keras")


    
    