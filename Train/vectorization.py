import tensorflow as tf
import numpy as np

from load_dataset import vocab

# Convert Characters Tokens to Character IDs
ids_from_chars = tf.keras.layers.StringLookup(vocabulary =list(vocab))

# Convert Character Tokens to Character IDs
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary= list(vocab), 
    invert = True
    )

# Get Text from ids
def text_from_ids(ids):
    return tf.strings.reduce_join(
        chars_from_ids(ids), 
        axis=-1
        )