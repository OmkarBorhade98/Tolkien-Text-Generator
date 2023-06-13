import tensorflow as tf

class MyModel(tf.keras.sequencial):
    """
    Constructor
    vocab_size = no of unique characters in dataset
    embeding_dim = output dimension for embedding layer
    rnn_nodes_list = list of rnn nodes in each layer
    """
    def __init__(self, vocab_size, embeding_dim, rnn_layer_node_list):
        super.__init__()
        # Adding Embeding Layer
        # Turns positive integers (indexes) into dense vectors of fixed size.
        self.add(tf.keras.layers.Embedding(
            input_dim = vocab_size, 
            output_dim = embeding_dim
        ))
        # Adding GRU Layers
        for index, units in enumerate(rnn_layer_node_list):
            self.add(tf.keras.layers.GRU(
                units = units,
                return_sequences= True,
                return_state= True
            ))
        # Adding Dense Layer
        self.add(tf.keras.layers.Dense(512))
        self.add(tf.keras.layers.Dense(vocab_size))
            
