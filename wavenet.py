import tensorflow as tf

init = tf.contrib.layers.xavier_initializer_conv2d()


def pad_input(value, dilation):
    '''
    Performs padding needed to make creation of dilation layer possible.
    Args:
        value: input to pad
        dilation: order of dilation value corresponds to
    '''
    shape = tf.shape(value)
    pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
    padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
    reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
    transposed = tf.transpose(reshaped, perm=[1, 0, 2])
    return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def pad_output(value, dilation):
    '''
    Pads value back to the original shape after convolution's transformation
    Args:
        value: input to reshape
        dilation: order of dilation value corresponds to
    '''
    shape = tf.shape(value)
    prepared = tf.reshape(value, [dilation, -1, shape[2]])
    transposed = tf.transpose(prepared, perm=[1, 0, 2])
    return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])

def pad_conv_output(value, transformed, dilation=1):
    '''
    Pads the output of dilated conv. to match and be evenly divisible by quant. channels.
    Args:
        value: input to pad
        transformed: the convolutional layer to pad to
    '''
    out_width = tf.shape(value)[1] - dilation
    return tf.slice(transformed, [0, 0, 0], [-1, out_width, -1])


def conv1d(value, kernel, pad=True):
    '''
    Creates a single convolutional layers, pads the input if necessary
    Args:
        value: input for the layer
        kernel: the filter used in convolution
    '''
    conv = tf.nn.conv1d(value, kernel, stride=1, padding='VALID')
    if pad:
        output = pad_conv_output(value, conv)
    else:
        output = conv
    return output


def dilated_conv(value, kernel, dilation):
    '''

    Creates a dilated convolutional layer
    '''

    transformed = pad_input(value, dilation)
    conv = tf.nn.conv1d(transformed, kernel, stride=1, padding='VALID')
    transformed = pad_output(conv, dilation)
    return pad_conv_output(value, transformed, dilation)


class Wavenet():
    """
    The WaveNet architecture, contains definitions for constructing and initializing WaveNet model.
    """
    def __init__(self, dilations, kernel_width, dilation_width, residual_width, skip_width, q_channels, receptive_field, log):
        '''
        Args:
            dilations: List of dilations used in WN
            kernel_width: filter with, first dimension of of kernel and gate dilation component
            dilation_width: width of dilation convolutional filters
            residual_width: width of residual convolutional filters
            skip_width: width of skip channels for the softmax output
            q_channels: number of quantization levels
            receptive_field: width of receptive field
            log: logger instance
        '''
        self.dilations = dilations
        self.kernel_w = kernel_width
        self.dil_w = dilation_width
        self.res_w = residual_width
        self.q_channels = q_channels
        self.skip_w = skip_width
        self.receptive_field = receptive_field
        self.keep_prob = tf.placeholder(tf.float32)
        variables = dict()
        with tf.variable_scope('wavenet_model'):
            variables['causal_layer'] = dict()
            variables['causal_layer']['kernel'] = tf.get_variable(name='kernel', shape=[self.kernel_w, self.q_channels, self.res_w], initializer=init)
            variables['dil_stack'] = []
            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('layer{}-dil{}'.format(i, dilation)):
                    current_layer = dict()
                    current_layer['kernel'] = tf.get_variable(name='kernel', shape=[self.kernel_w, self.res_w, self.dil_w], initializer=init)
                    current_layer['gate'] = tf.get_variable(name='gate', shape=[self.kernel_w, self.res_w, self.dil_w], initializer=init)
                    current_layer['dense'] = tf.get_variable(name='dense', shape=[1, self.dil_w, self.res_w], initializer=init)
                    current_layer['skip'] = tf.get_variable(name='skip', shape=[1, self.dil_w, self.skip_w], initializer=init)
                    variables['dil_stack'].append(current_layer)

            prostproc_layer = dict()
            prostproc_layer['pp1'] = tf.get_variable(name='pp1', shape=[1, self.skip_w, self.skip_w], initializer=init)
            prostproc_layer['pp2'] = tf.get_variable(name='pp2', shape=[1, self.skip_w, self.q_channels], initializer=init)
            
            variables['pp'] = prostproc_layer

        self.variables = variables


    def construct_network(self, network_input):
        '''
        Initializes the network as a stack of dilations with pre- and post-processing layers.
        Goes over the list of orders of dilations and creates a convolution a trous for every one.
        Then, a series of post-processing operations is used and the output
        Args:
            network_input: pre-processed audio input
        '''
        network_input = tf.slice(network_input, [0, 0, 0], [-1, tf.shape(network_input)[1]-1, -1])
        current_l = conv1d(network_input, self.variables['causal_layer']['kernel'])
        final_w = tf.shape(network_input)[1] - self.receptive_field + 1
        
        outputs = []
        for i, dilation in enumerate(self.dilations):
            if dilation == 1:
                conv_w = conv1d(current_l, self.variables['dil_stack'][i]['kernel'])
                conv_g = conv1d(current_l, self.variables['dil_stack'][i]['gate'])
            else:
                conv_w = dilated_conv(current_l, self.variables['dil_stack'][i]['kernel'], dilation)
                conv_g = dilated_conv(current_l, self.variables['dil_stack'][i]['gate'], dilation)

            out = tf.tanh(conv_w) * tf.sigmoid(conv_g)

            out_conv = conv1d(out, self.variables['dil_stack'][i]['dense'], False)
            
            out_skip = tf.slice(out, [0, tf.shape(out)[1] - final_w, 0], [-1, -1, -1])
            skip = conv1d(out_skip, self.variables['dil_stack'][i]['skip'], False)
            
            input_cut = tf.shape(current_l)[1] - tf.shape(out_conv)[1]
            input = tf.slice(current_l, [0, input_cut, 0], [-1, -1, -1])

            current_l = input + out_conv
            outputs.append(skip)

        total = sum(outputs)
        out_conv = conv1d(tf.nn.relu(total), self.variables['pp']['pp1'], False)

        relu = tf.nn.relu(out_conv)
        drop_out = tf.nn.dropout(relu, self.keep_prob)
        out_conv = conv1d(drop_out, self.variables['pp']['pp2'], False)
        return out_conv