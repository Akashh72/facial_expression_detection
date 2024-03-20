import tensorflow as tf
from tensorflow.keras import layers


def base_model_generator():
    #creating base model
    def build_model(num_classes):
        inputs = layers.Input(shape=(224, 224, 3))
        model = tf.keras.applications.VGG19(include_top=False, input_tensor=inputs, weights="imagenet")
        # Freeze the pretrained weights
        model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

        # Compile
        model = tf.keras.Model(inputs, outputs, name="VGG19")
        return model