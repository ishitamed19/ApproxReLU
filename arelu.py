import tensorflow as tf
from keras import backend as K

class ApproxRelu(Layer):
    
    def __init__(self, k=0.5, n=1.1, **kwargs):
        super(ApproxRelu, self).__init__(**kwargs)
        self.supports_masking = True
        self.k = K.cast_to_floatx(k)
        self.n = K.cast_to_floatx(n)

    def call(self, inputs):
        orig = inputs
        inputs = tf.where(orig <= 0.0, tf.zeros_like(inputs), inputs)
        inputs = tf.where(tf.greater(orig, 0.0), self.k*tf.pow(inputs, self.n), inputs)
        return  inputs

    def get_config(self):
        config = {'k': float(self.k), 'n': float(self.n)}
        base_config = super(ApproxRelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
