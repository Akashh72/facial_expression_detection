import tensorflow as tf

def finetune(base_model_path,fine_tune_layer_number):
    model = tf.keras.models.load_model(base_model_path)
    model.trainable = True

    for layer in model.layers[:fine_tune_layer_number]:
        layer.trainable = False
    
    #Default the batch normalization layer is freez
    model.layers[-3].trainable = False

    return model