import tensorflow as tf


class QNetwork(tf.keras.Model):
    """Actor (Policy) Model.

    This class construct the model.
    """

    def __init__(self, state_size, action_size, d1_units=64, d2_units=64):
        """ Initialize parameters and build model.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            d1_units: Integer. Number of nodes in first hidden layer
            d2_units: Integer. Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.d1 = tf.keras.layers.Dense(d1_units, activation='relu', name='Dense_1', input_shape=(state_size,))
        self.d2 = tf.keras.layers.Dense(d2_units, activation='relu', name='Dense_2')
        self.d3 = tf.keras.layers.Dense(action_size, name='output')

    def __repr__(self):
        return 'QNetwork'

    def __str__(self):
        return 'QNetwork'

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        """Calls the model on new inputs.

        Args:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to
                run the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be either a tensor or
                None (no mask).:

        Returns:
            A tensor if there is a single output, or a list of tensors if there
                are more than one outputs.
        """
        # Define the hidden layers
        hidden = self.d1(inputs)
        hidden = self.d2(hidden)

        return self.d3(hidden)
