import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# --- Configuration ---
# Define the paths to your dataset
TRAIN_DIR = 'dataset/training'
VALID_DIR = 'dataset/validation'

# Define image and training parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10 # Start with 10 and increase if needed
LEARNING_RATE = 0.001

# --- 1. Prepare the Data ---
# We use ImageDataGenerator to load images from the directories and apply data augmentation
# to the training set to prevent overfitting and improve generalization.

print("Setting up data generators...")

# Data augmentation configuration for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Rescale pixel values from [0, 255] to [0, 1]
    rotation_range=40,          # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,      # Randomly shift images horizontally
    height_shift_range=0.2,     # Randomly shift images vertically
    shear_range=0.2,            # Apply shear transformations
    zoom_range=0.2,             # Randomly zoom in on images
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Strategy for filling in newly created pixels
)

# For the validation set, we only need to rescale the pixel values.
# We do not apply augmentation here to get a true measure of performance.
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators that will read images from the source directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes (breeds) from the generator
num_classes = len(train_generator.class_indices)
print(f"Found {num_classes} classes (breeds) to train.")
print("Class labels:", train_generator.class_indices)


# --- 2. Build the Model (Transfer Learning) ---

print("Building model with MobileNetV2 base...")

# Load the MobileNetV2 model, pre-trained on ImageNet.
# We exclude the final classification layer (`include_top=False`)
# because we will create our own.
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Freeze the convolutional base. This is the key to transfer learning.
# We don't want to update the learned features of the base model.
base_model.trainable = False

# Create our new model "head" to place on top of the frozen base
x = base_model.output
x = GlobalAveragePooling2D()(x) # Pool the features to a single vector
x = Dense(1024, activation='relu')(x) # A fully-connected layer for learning
# The final classification layer. It must have as many neurons as we have classes.
# 'softmax' is used for multi-class probability outputs.
predictions = Dense(num_classes, activation='softmax')(x)

# This is the final model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# --- 3. Compile the Model ---
# We configure the model for training.

print("Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy', # Standard loss function for multi-class classification
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()


# --- 4. Train the Model ---

print(f"Starting training for {EPOCHS} epochs...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    # Calculate steps per epoch based on dataset size and batch size
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

print("Training complete.")


# --- 5. Save the Final Model ---
# The saved model file contains the architecture, weights, and training config.

output_model_path = 'breed_classifier_model.h5'
print(f"Saving trained model to {output_model_path}...")
model.save(output_model_path)
print("Model saved successfully. You can now use this file in your Streamlit app.")

