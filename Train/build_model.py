import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Constructor
    vocab_size = no of unique characters in dataset
    embeding_dim = output dimension for embedding layer
    rnn_nodes_list = list of rnn nodes in each layer
    """
    def __init__(self, vocab_size, embeding_dim, rnn_layer_node_list):
        super().__init__()
        # Adding Embeding Layer
        # Turns positive integers (indexes) into dense vectors of fixed size.
        self.embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size, 
            output_dim = embeding_dim
        )
        # Adding GRU Layers
        self.gru = []
        for units in rnn_layer_node_list:
            self.gru.append(tf.keras.layers.GRU(
                units = units,
                return_sequences= True,
                return_state= True
            ))
        # Adding Dense Layer
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training = False):
        x= inputs
        x= self.embedding(x, training = training)
        for index, gru_layer in enumerate(self.gru):
            if states is None:
                states = self.gru[index].get_initial_state(x)
            x, states = self.gru[index](x, initial_state =states, training =training)
        x = self.dense(x, training = training)
        if return_state:
            return x, states
        else:
            return x
