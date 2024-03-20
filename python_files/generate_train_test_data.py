import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def data_generator(dir,batch,img_size,seed):
    BATCH_SIZE = batch
    IMG_SIZE = img_size

    dataset_dir = dir

    class_names = os.listdir(dataset_dir)

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int", 
        class_names=class_names,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=seed,
        color_mode="grayscale"
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int", 
        class_names=class_names,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        color_mode="grayscale"
    )

    print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))

    print("Label Mode is in integer so the loss function preferable sparse categorical")

    # Define your data augmentation pipeline
    train_data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03),      # Random rotation by up to 20 degrees
        tf.keras.layers.RandomFlip("horizontal"),  # Random horizontal flip
        tf.keras.layers.Rescaling(1./255),
    ])

    validation_data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
    ])
    def preprocess_image(image, label):
        # Convert grayscale images to RGB
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)
        # Resize images to a uniform size
        image = tf.image.resize(image, [224, 224])
        
        return image, label
    

    # Apply data augmentation to the training dataset
    train_augmented_dataset = train_dataset.map(lambda x, y: (train_data_augmentation(x), y))
    train_augmented_dataset = train_augmented_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    valid_augmented_dataset = validation_dataset.map(lambda x, y: (validation_data_augmentation(x), y))
    valid_augmented_dataset = valid_augmented_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_augmented_dataset,valid_augmented_dataset,class_names