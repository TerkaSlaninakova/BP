import tensorflow as tf

init = tf.contrib.layers.xavier_initializer_conv2d()

def pad_input(value, dilation):
    '''
    Performs padding needed to make creation of dilation layer possible
    '''
    shape = tf.shape(value)
    pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
    padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
    reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
    transposed = tf.transpose(reshaped, perm=[1, 0, 2])
    return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

def pad_output(value, dilation):
    '''
    Pads back to the original shape
    '''
    shape = tf.shape(value)
    prepared = tf.reshape(value, [dilation, -1, shape[2]])
    transposed = tf.transpose(prepared, perm=[1, 0, 2])
    return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])

def pad_conv_output(value, transformed, kernel_width, dilation=1):
    '''
    Pads the output of dilated conv. to match and be evenly divisible by quant. channels
    '''
    out_width = tf.shape(value)[1] - (kernel_width - 1) * dilation
    return tf.slice(transformed, [0, 0, 0], [-1, out_width, -1])

def conv1d(value, kernel, pad=True):
    conv = tf.nn.conv1d(value, kernel, stride=1, padding='VALID')
    if pad:
        output = pad_conv_output(value, conv, tf.shape(kernel)[0])
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
    return pad_conv_output(value, transformed, tf.shape(kernel)[0], dilation)

class Wavenet():

    def __init__(self, dilations, kernel_width, dilation_width, residual_width, skip_width, q_channels, receptive_field, log):
        self.dilations = dilations
        self.kernel_w = kernel_width
        self.dil_w = dilation_width
        self.res_w = residual_width
        self.q_channels = q_channels
        self.skip_w = skip_width
        self.receptive_field = receptive_field

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
        out_conv = conv1d(tf.nn.relu(out_conv), self.variables['pp']['pp2'], False)
        return out_conv