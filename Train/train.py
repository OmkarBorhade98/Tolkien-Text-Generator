import numpy as np
import tensorflow as tf

from load_dataset import dataset_text
from load_dataset import vocab
from load_dataset import split_input_target
import vectorization
import build_model

# Converting Text to array of character IDs
all_ids = vectorization.ids_from_chars(
    tf.strings.unicode_split(
        input = dataset_text, 
        input_encoding = "UTF-8"
    )
)

# 
ids_datasets = tf.data.Dataset.from_tensor_slices(all_ids)
print("ids_dataset shape:", tf.shape(ids_datasets))

seq_length = 100
examples_per_epoch = len(dataset_text) // (seq_length + 1)

sequences = ids_datasets.batch(
    batch_size= seq_length +1, 
    drop_remainder = True
)

# Creating Target and Test Datasets
dataset = sequences.map(split_input_target)
print("dataset shape:", dataset.numpy().shape())


# Creating Batch Dataset
BATCH_SIZE = 512
BUFFER_SIZE = 10000
dataset = (
    dataset.shuffle(buffer_size = BUFFER_SIZE).
    batch(batch_size = BATCH_SIZE,
        drop_remainder = True
    ).
    prefetch(
        tf.data.experimental.AUTOTUNE
    )
)

embeding_dim = 256
rnn_layer_node_list = [512, 512, 512]

model = build_model.MyModel(
    vocab_size = vectorization.ids_from_chars.vocabulary_size(),
    embeding_dim = embeding_dim,
    rnn_layer_node_list = rnn_layer_node_list
)

#Print Model Summarypython trai9n
model.summary()

