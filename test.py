import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("C:/Zeeshan/trained images/converted_keras/keras_model.h5")

# Check for DepthwiseConv2D layers and update them if necessary
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        layer_config = layer.get_config()
        layer_config.pop("groups", None)
        updated_layer = tf.keras.layers.DepthwiseConv2D.from_config(layer_config)
        model = tf.keras.models.clone_model(model, clone_function=lambda l: updated_layer if l == layer else l)

# Save the model
model.save("new_model_path", save_format="tf")

# Optionally suppress warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
