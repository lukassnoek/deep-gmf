""" An implementation of a recurrent BLT (bottom-up, lateral, top-down)
network, as in e.g.:

Thorat et al. (2021). Category-orthogonal object features guide
information processing in recurrent neural networks
trained for object categorization. SVRHM 2021 at Neurips, 2021.

Just a proof of concept, not used in the deep-gmf project (at this moment).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Conv2DTranspose
from tensorflow.keras.layers import Layer, Flatten, Reshape, Lambda


class BLTDenseLayer(Layer):
    
    def __init__(self, input_shape, units, *args, batch_size=256, return_sequences=True, T=5, **kwargs):
        super().__init__(*args, **kwargs)
        # Need to know input_shape to correctly initialize state
        self.input_shape_ = (batch_size,) + input_shape
        self.batch_size = batch_size
        self.return_sequences = return_sequences
        self.flatten = Flatten()
        self.dense = Dense(units=units)

        state_shape = (self.batch_size, units)  # dense state
        self.state = tf.Variable(tf.zeros(state_shape), shape=state_shape, trainable=False)
        self.new_state = tf.Variable(tf.zeros(state_shape), shape=state_shape, trainable=False)
        
        if self.return_sequences:
            # This does not work yet! For some reason, it doesn't update self.sequences in self.call
            self.t = 0
            self.sequences = [tf.Variable(tf.zeros(state_shape), shape=state_shape, trainable=False)
                               for i in range(T+1)]

    def call(self, x):
        x, _ = x  # discard top-down info (0.0)
        x = self.flatten(x)
        out = self.dense(x)
        self.new_state.assign(out)

        if self.return_sequences:
            # Quite memory heavy, so only do when requested
            self.sequences[self.t].assign(out)
            self.t = self.t + 1
    
        return out, self.sequences

    def flip_state(self):
        self.state.assign(self.new_state)


class BLTConvLayer(Layer):
    
    def __init__(self, input_shape, filters, *args, batch_size=256, return_sequences=True, T=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape_ = (batch_size,) + input_shape
        self.batch_size = batch_size
        self.conv = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same', activation='relu')
        self.return_sequences = return_sequences
        
        # Compute output shape of conv layer to determine the shape of the state
        self.output_shape_ = self.conv.compute_output_shape(self.input_shape_)
        self.state = tf.Variable(tf.zeros(self.output_shape_), shape=self.output_shape_, trainable=False)
        self.new_state = tf.Variable(tf.zeros(self.output_shape_), shape=self.output_shape_, trainable=False)
        
        if self.return_sequences:
            self.t = 0
            self.sequences = [tf.Variable(tf.zeros(self.output_shape_), shape=self.output_shape_, trainable=False)
                              for i in range(T+1)]

    def build(self, input_shapes):
        
        if input_shapes[0][1:] != self.input_shape_[1:]:
            expected = self.input_shape_[1:]
            actual = input_shape[0][1:]
            raise ValueError(f"Expected input shape {expected} does not match actual input shape {actual}!")
        
        # Depending on how the top-down input looks like, initialize a feedback layer
        _, td_inp_shape = input_shapes
        if td_inp_shape is None:
            # No feedback! Just pass input (0.0)
            self.feedback_td = Lambda(lambda x: x)
        elif len(td_inp_shape) > 2:
            # Transposed convolution to upsample top-down info to current layer
            stride = self.output_shape_[1] // td_inp_shape[1]
            self.feedback_td = Conv2DTranspose(self.output_shape_[-1], self.conv.kernel_size,
                                               strides=stride, padding='same', activation='relu')
        else:
            # A densely connected layer to map dense top-down info to previous conv layer
            units = tf.reduce_prod(self.output_shape_[1:])
            dense = Dense(units=units, activation='relu')
            reshape = Reshape(self.output_shape_[1:])
            self.feedback_td = lambda x: reshape(dense(x))
            
    def call(self, inputs):
        bu, td = inputs  # bottom-up, top-down
        x = self.conv(bu)
        td = self.feedback_td(td)
        out = x * (1 + self.state + td)
        self.new_state.assign(out)
        
        if self.return_sequences:
            self.sequences[self.t].assign(out) 
            self.t = self.t + 1
        
        #tf.print(self.sequences)
        return out, self.sequences

    def flip_state(self):
        self.state.assign(self.new_state)


def BLTNet(input_shape=(32, 32, 3), batch_size=256, biological_unrolling=True):

    inp_ = Input(shape=input_shape)
    n_filters = [32, 64, 128, 256]

    layers = [None]
    current_shape = input_shape
    for i, nf in enumerate(n_filters):  # create layers!
        # Stride is 2, so downsample with factor two in H and W, but channels (filters) * 2
        conv = BLTConvLayer(input_shape=tuple(current_shape), filters=nf, batch_size=batch_size, name=f'conv_{i}')
        layers.append(conv)
        
        # Downsample shape 
        current_shape = [s // 2 for s in current_shape[:-1]]
        current_shape = current_shape + [nf]

    # Add dense layer
    dense = BLTDenseLayer(units=1, input_shape=tuple(current_shape), batch_size=batch_size, name='flatten_dense')
    layers.append(dense)
    
    L = len(layers)
    T = 5

    # io_ contains the in/output from all layers, including the dense one
    io_ = [inp_, None, None, None, None, None]
    for t in range(1, T+1):  # loop over timepoints 1 ... T
        if biological_unrolling:
            # Only the first `t` layers can be accessed at timepoint `t`!
            # E.g., at timepoint 1, only layer 1 is used (inp -> L1), at 
            # timepoint 2, both layer 1 (inp -> L1) and layer 2 (L1 -> L2)
            # are accessed, etc. Up to L layers
            iter_ = range(1, min(t+1, L))
        else:
            iter_ = range(1, L)

        for i in iter_:
            layer = layers[i]  # current layer
            bu = io_[i-1]  # bottom-up

            # Dense layer does not get top-down info, so set to 0.
            if i == (L - 1): 
                td = 0.
            else:
                # Regular conv layers get top-down info from layer l+1
                # Difference from Thorat et al. is that it only gets top-down
                # info from immediately following layer (i+1)
                td = layers[i+1].state

            # Store result in io_ list; note that the state is saved internally
            # (as self.new_state, to be updated to self.state after the layer-loop)
            out = layer([bu, td])
            io_[i] = out[0]

        for i in range(1, min(t+1, L)):
            # After every timepoint, the states should be
            # updated (new_state -> state)
            layers[i].flip_state()

    y = io_[-1]  # final prediction is dense output after last timepoint
    model = Model(inputs=inp_, outputs=y, name='BLTNet')
    return model
