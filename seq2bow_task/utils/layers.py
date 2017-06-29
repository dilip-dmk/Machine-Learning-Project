from __future__ import absolute_import
from __future__ import division
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils

import numpy as np
import itertools

from classifiers import *


#### Initializers ####
def kaiming_initializer():
    """
    https://arxiv.org/pdf/1502.01852v1.pdf
    """
    return tf.contrib.layers.variance_scaling_initializer()

def constant_initializer(x):
    return tf.constant_initializer(x)

class Initializer(object):
    def __call__(self, shape, dtype=None, partition_info=None):
        raise NotImplementedError

#### block by NickPAN ####
class Orthogonal_Initializer(Initializer):
    def __init__(self, gain=1.0, dtype=tf.float32, seed=None):
        self.gain = gain
        self.dtype = dtype
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        # Check the shape
        if len(shape) < 2:
            raise ValueError("The tensor to initialize must be at least two-dimensional")
        # Flatten the input shape with the last dimension remaining
        # its original shape so it works for conv2d
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)

        # Generate a random matrix
        a = tf.random_uniform(flat_shape, dtype=dtype, seed=self.seed)
        # Compute the svd
        _, u, v = tf.svd(a, full_matrices=False)
        # Pick the appropriate singular value decomposition
        if num_rows > num_cols:
            q = u
        else:
            # Tensorflow departs from numpy conventions
            # such that we need to transpose axes here
            q = tf.transpose(v)
        return self.gain * tf.reshape(q, shape)


#### Layers ####
class BasicLayer(object):
    """
    Abstract class for layers.
    """
    def __init__(self,
                 in_dim=None,
                 dim=None,
                 name=None):
        self.in_dim = in_dim
        self.dim = dim
        self.name = name
        self.set_variables()

    def get_output(self, **inputs):
        return self.build_layer(**inputs)

    @abstractmethod
    def set_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def build_layer(self, **inputs):
        raise NotImplementedError()


#### block by NickPAN ####
class AttentionLayer(BasicLayer):
    """
    Attention layer.
    -Args:
        fan_in: A tensor with shape [batch_size, num_steps, in_dim].
    -Return:
        atn_out: A tensor with shape [batch_size, in_dim].
    """
    def __init__(self,
                 name,
                 in_dim,
                 dim,
                 num_steps,
                 initializer=None):
        self.num_steps = num_steps
        self.initializer = initializer
        super(self.__class__, self).__init__(in_dim=in_dim,
                                             dim=dim,
                                             name=name)

    def set_variables(self):
        in_dim = self.in_dim
        dim = self.dim
        with tf.variable_scope(self.name, initializer=self.initializer):
            atn_W = tf.get_variable("attention_W",
                shape=[1, 1, in_dim, dim])
            atn_b = tf.get_variable("attention_b",
                shape=[dim])
            atn_v = tf.get_variable("attention_v",
                shape=[1, 1, dim, 1])

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]
        _fan_in = tf.expand_dims(fan_in, 2)  # [batch_size, num_steps, 1, in_dim]
        if "name" in inputs:
            name = inputs["name"]
        else:
            name = None


        with tf.variable_scope(self.name, reuse=True):
            atn_W = tf.get_variable("attention_W")
            atn_b = tf.get_variable("attention_b")
            atn_v = tf.get_variable("attention_v")
            activations = tf.nn.tanh(tf.nn.conv2d(_fan_in, atn_W,
                strides=[1, 1, 1, 1], padding="SAME") + atn_b)
            scores = tf.nn.conv2d(activations, atn_v, strides=[1, 1, 1, 1],
                padding="SAME")  # [batch_size, num_steps, 1, 1]
            atn_probs = tf.nn.softmax(tf.squeeze(scores, [2, 3]))  # [batch_size, num_steps]
            _atn_out = tf.batch_matmul(tf.expand_dims(atn_probs, 1), fan_in)  # [batch_size, 1, dim]
            atn_out = tf.squeeze(_atn_out, [1], name=name)

        return atn_out


#### block by NickPAN ####
class Dynamic_AttentionLayer(BasicLayer):
    """
    Attention layer.
    Args:
        -fan_in: A tensor with shape [batch_size, num_steps, in_dim].
    Return:
        -atn_out: A tensor with shape [batch_size, in_dim].
    """
    def __init__(self,
                 name,
                 in_dim,
                 dim,
                 num_steps,
                 batch_size,
                 initializer=None):
        self.num_steps = num_steps
        self.initializer = initializer
        self.batch_size = batch_size
        super(self.__class__, self).__init__(in_dim=in_dim,
                                             dim=dim,
                                             name=name)

    def set_variables(self):
        in_dim = self.in_dim
        dim = self.dim
        with tf.variable_scope(self.name, initializer=self.initializer):
            atn_W = tf.get_variable("attention_W",
                                    shape=[1, 1, in_dim, dim])
            atn_b = tf.get_variable("attention_b",
                                    shape=[dim])
            atn_v = tf.get_variable("attention_v",
                                    shape=[1, 1, dim, 1])

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]
        _length = inputs["lengths"]
        batch_size, num_steps = self.batch_size, self.num_steps
        length = tf.slice(_length, [0], [batch_size])
        # [batch_size, num_steps, 1, in_dim]
        _fan_in = tf.expand_dims(fan_in, 2)


        with tf.variable_scope(self.name, reuse=True):
            atn_W = tf.get_variable("attention_W")
            atn_b = tf.get_variable("attention_b")
            atn_v = tf.get_variable("attention_v")
            activations = tf.nn.tanh(
                tf.nn.conv2d(_fan_in, atn_W,
                             strides=[1, 1, 1, 1],
                             padding="SAME")
                + atn_b)
            # [batch_size, num_steps, 1, 1]
            scores = tf.nn.conv2d(activations,
                                  atn_v,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")
            _idx = tf.tile([tf.range(num_steps)], [batch_size, 1])
            mask = tf.less(_idx, tf.expand_dims(length, 1))
            zeros = tf.zeros([batch_size, num_steps])
            mask_logits = tf.where(mask, tf.squeeze(scores, [2, 3]), zeros)
            exp_logits = tf.exp(mask_logits
                                - tf.reduce_max(mask_logits,
                                                axis=1,
                                                keep_dims=True)
                                )
            mask_exps = tf.where(mask, exp_logits, zeros)
            atn_probs = tf.div(
                mask_exps,
                tf.reduce_sum(mask_exps, axis=1, keep_dims=True)
            )

            # [batch_size, num_steps]
            # atn_probs = tf.nn.softmax(tf.squeeze(scores, [2, 3]))

            # [batch_size, 1, in_dim]
            _atn_out = tf.batch_matmul(tf.expand_dims(atn_probs, 1),
                                       fan_in)
            atn_out = tf.squeeze(_atn_out, [1])

        return atn_out


#### block by NickPAN ####
class FCLayer(BasicLayer):
    """
    Fully connected layer.
    Args:
        -fan_in: A tensor with shape [batch_size, in_dim].
        -activation_fn: A tensorflow activation function.
    -Return:
        -fc_out: A tensor with shape [batch_size, dim]
    """
    def __init__(self,
                 name,
                 in_dim,
                 dim,
                 activation_fn,
                 initializer=None):
        self.activation_fn = activation_fn
        self.initializer = initializer
        super(self.__class__, self).__init__(in_dim=in_dim,
                                             dim=dim,
                                             name=name)

    def set_variables(self):
        in_dim = self.in_dim
        dim = self.dim
        with tf.variable_scope(self.name, initializer=self.initializer):
            fc_W = tf.get_variable("fc_W", shape=[in_dim, dim])
            fc_b = tf.get_variable("fc_b", shape=[dim])

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]

        with tf.variable_scope(self.name, reuse=True):
            fc_W = tf.get_variable("fc_W")
            fc_b = tf.get_variable("fc_b")
            fc_out = self.activation_fn(tf.matmul(fan_in, fc_W) + fc_b)

            l2_loss = tf.nn.l2_loss(fc_W)

        return (fc_out, l2_loss)


#### block by NickPAN ####
class RNNBatchnromLayer(BasicLayer):
    """
    RNNBatchnorm Layer.
    Args:
        -fan_in: A tensor to be batchnormed.
        -is_training: 0D bool tensor, control the update of running mean and
            output of batchnorme.
        -decay: float, decay of running mean.
    Return:
        -bn_out: A tensor has same shape as x.
    """
    def __init__(self, name, center=True, scale=True, decay=0.9):
        self.decay = decay
        self.center = center
        self.scale = scale
        self.initializers = {"gamma": tf.constant_initializer(0.1)}
        super(self.__class__, self).__init__(name=name)

    def set_variables(self):
        tf.no_op()

    def _bn_in_train(self):
        with tf.variable_scope(self.name) as bn_scope:
            return tf.contrib.layers.batch_norm(
                self.fan_in, decay=self.decay, center=self.center,
                scale=self.scale, param_initializers=self.initializers,
                scope=bn_scope, is_training=True, reuse=None,
                updates_collections=None)

    def _bn_in_test(self):
        with tf.variable_scope(self.name) as bn_scope:
            return tf.contrib.layers.batch_norm(
                self.fan_in, decay=self.decay, center=self.center,
                scale=self.scale, param_initializers=self.initializers,
                scope=bn_scope, is_training=False, reuse=True,
                updates_collections=None)

    def build_layer(self, **inputs):
        self.fan_in = inputs["fan_in"]
        is_training = inputs["is_training"]

        bn_out = tf.cond(is_training, self._bn_in_train, self._bn_in_test)

        return bn_out


#### block by NickPAN ####
class BatchnromLayer(BasicLayer):
    """
    Batchnorm Layer
    Args:
        -fan_in: A tensor to be batchnormed.
        -is_training: 0D bool tensor, control the update of running mean and
            output of batchnorme.
        -decay: float, decay of running mean.
    Return:
        -bn_out: A tensor has same shape as x.
    """
    def __init__(self, name, center=True, scale=True, decay=0.9):
        self.decay = decay
        self.center = center
        self.scale = scale
        super(self.__class__, self).__init__(name=name)

    def set_variables(self):
        tf.no_op()

    def _bn_in_train(self):
        with tf.variable_scope(self.name) as bn_scope:
            return tf.contrib.layers.batch_norm(
                                    self.fan_in,
                                    decay=self.decay,
                                    center=self.center,
                                    scale=self.scale,
                                    scope=bn_scope,
                                    is_training=True,
                                    reuse=None,
                                    updates_collections=None)

    def _bn_in_test(self):
        with tf.variable_scope(self.name) as bn_scope:
            return tf.contrib.layers.batch_norm(
                                    self.fan_in,
                                    decay=self.decay,
                                    center=self.center,
                                    scale=self.scale,
                                    scope=bn_scope,
                                    is_training=False,
                                    reuse=True,
                                    updates_collections=None)

    def build_layer(self, **inputs):
        self.fan_in = inputs["fan_in"]
        is_training = inputs["is_training"]

        bn_out = tf.cond(is_training, self._bn_in_train, self._bn_in_test)

        return bn_out


#### block by NickPAN ####
class ResGLULayer(BasicLayer):
    """
    A bottleneck structure pre-activated residual block of GLU.
    http://arxiv.org/abs/1612.08083
    Args:
        -fan_in: A tensor with shape [batch_size, in_height, in_width, in_dim].
        -in_dim: int, dims of fan_in.
        -out_dim: int, dims of fan_out.
        -filter_width: int, width of filter.
        -dim: int, hidden_size of filter.
        -is_training: 0D bool tensor, control the mod of batchnorm.
        -stride: int, stride of filter.
        -name: str, name of this op.
    Return:
        -glu_out: A tensor with shape
            [batch_size, out_height, out_width, output_size].
        -l2_loss: A tensor with 0D, l2 regulization of weights
    """
    def __init__(self,
                 name,
                 in_dim,
                 out_dim,
                 filter_width,
                 dim,
                 stride=1):
        self.filter_width = filter_width
        self.out_dim = out_dim
        self.stride = stride
        super(self.__class__, self).__init__(in_dim=in_dim,
                                             dim=dim,
                                             name=name)

    def set_variables(self):
        in_dim = self.in_dim
        out_dim = self.out_dim
        filter_width = self.filter_width
        stride = self.stride
        dim = self.dim

        with tf.variable_scope(self.name):
            prev_filter = tf.get_variable(
                "prev_filter",
                shape=[1, 1, in_dim, dim],
                initializer=kaiming_initializer())
            prev_bias = tf.get_variable(
                "prev_bias",
                shape=[dim],
                initializer=constant_initializer(0))

            conv_filter = tf.get_variable(
                "conv_filter",
                shape=[1, filter_width, dim, dim],
                initializer=kaiming_initializer())
            conv_bias = tf.get_variable(
                "conv_bias",
                shape=[dim],
                initializer=constant_initializer(0))

            gate_filter = tf.get_variable(
                "gate_filter",
                shape=[1, filter_width, dim, dim],
                initializer=kaiming_initializer())
            gate_bias = tf.get_variable(
                "gate_bias",
                shape=[dim],
                initializer=constant_initializer(0))

            post_filter = tf.get_variable(
                "post_filter",
                shape=[1, 1, dim, out_dim],
                initializer=kaiming_initializer())
            post_bias = tf.get_variable(
                "post_bias",
                shape=[out_dim],
                initializer=constant_initializer(0))

            self.bn_layer_1 = BatchnromLayer("batch_norm_layer_1")
            self.bn_layer_2 = BatchnromLayer("batch_norm_layer_2")

            if stride != 1 or in_dim != out_dim:  # projection of fan_in
                self.bn_layer_3 = BatchnromLayer("batch_norm_layer_3")
                proj_filter = tf.get_variable(
                    "proj_filter",
                    shape=[1, 1, in_dim, out_dim])
                proj_bias = tf.get_variable(
                    "proj_bias",
                    shape=[out_dim])

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]
        is_training = inputs["is_training"]
        # dropout
        if inputs.has_key("keep_prob"):
            keep_prob = inputs["keep_prob"]
        else:
            keep_prob = 1.
        stride = self.stride
        in_dim = self.in_dim
        out_dim = self.out_dim

        with tf.variable_scope(self.name, reuse=True):
            prev_filter = tf.get_variable("prev_filter")
            prev_bias = tf.get_variable("prev_bias")

            conv_filter = tf.get_variable("conv_filter")
            conv_bias = tf.get_variable("conv_bias")

            gate_filter = tf.get_variable("gate_filter")
            gate_bias = tf.get_variable("gate_bias")

            post_filter = tf.get_variable("post_filter")
            post_bias = tf.get_variable("post_bias")

        with tf.name_scope(self.name):
            l2_loss = tf.nn.l2_loss(prev_filter) + \
                    tf.nn.l2_loss(conv_filter) + \
                    tf.nn.l2_loss(gate_filter) + \
                    tf.nn.l2_loss(post_filter)

            bn_layer_1_name = self.name + "/" + "batch_norm_layer_1"
            bn_layer_1 = BatchnromLayer(bn_layer_1_name)
            _prev = bn_layer_1.get_output(fan_in=fan_in,
                    is_training=is_training)
            prev_activation = tf.nn.dropout(tf.nn.relu(_prev), keep_prob)
            prev = tf.nn.conv2d(prev_activation, prev_filter,
                    strides=[1, 1, 1, 1], padding="SAME") + prev_bias

            _linear = tf.nn.conv2d(prev, conv_filter,
                    strides=[1, 1, stride, 1], padding="SAME") + conv_bias
            _gate = tf.nn.conv2d(prev, gate_filter,
                    strides=[1, 1, stride, 1], padding="SAME") + gate_bias
            glu = _linear * tf.nn.sigmoid(_gate)

            bn_layer_2_name = self.name + "/" + "batch_norm_layer_2"
            bn_layer_2 = BatchnromLayer(bn_layer_2_name)
            _post = bn_layer_2.get_output(fan_in=glu,
                    is_training=is_training)
            post_activation = tf.nn.dropout(tf.nn.relu(_post), keep_prob)
            post = tf.nn.conv2d(tf.nn.relu(post_activation), post_filter,
                    strides=[1, 1, 1, 1], padding="SAME") + post_bias

            if stride != 1 or in_dim != out_dim:  # projection of fan_in
                with tf.variable_scope(self.name, reuse=True):
                    proj_filter = tf.get_variable("proj_filter")
                    proj_bias = tf.get_variable("proj_bias")

                bn_layer_3_name = self.name + "/" + "batch_norm_layer_3"
                bn_layer_3 = BatchnromLayer(bn_layer_3_name)
                _proj_in = bn_layer_3.get_output(fan_in=fan_in,
                    is_training=is_training)
                proj_in = tf.nn.conv2d(_proj_in, proj_filter,
                    strides=[1, 1, stride, 1], padding="SAME") + proj_bias
                glu_out = proj_in + post
            else:
                glu_out = fan_in + post

        return (glu_out, l2_loss)


#### block by NickPAN ####
class ResGLUWrapper(object):
    """
    Wapper for ResGLULayer.
    Args:
        -num_blocks: int, number of ResGLU blocks.
        -in_dim: int, in_dim of first ResGLULayer.
        -paras: list of tuples, with len(paras) = num_blocks; each tuple should
            be (dim, out_dim, stride, filter_width) of a ResGLULayer.
    """
    def __init__(self, num_blocks, in_dim, paras):
        if num_blocks != len(paras):
            raise ValueError("num_blocks should match length of paras")
        self.glu_layers = []
        self.num_blocks = num_blocks
        self.in_dim = in_dim
        self.paras = paras

        with tf.name_scope("ResGLUWrapper"):
            last_out_dim = 0
            for i, t in zip(range(num_blocks), paras):
                name = "glu_layer_%s" %str(i)
                # first glu layer
                if i == 0:
                    glu_layer = ResGLULayer(name,
                                            in_dim=self.in_dim,
                                            out_dim=t[1],
                                            filter_width=t[3],
                                            dim=t[0],
                                            stride=t[2])
                    last_out_dim = t[1]
                else:
                    glu_layer = ResGLULayer(name,
                                            in_dim=last_out_dim,
                                            out_dim=t[1],
                                            filter_width=t[3],
                                            dim=t[0],
                                            stride=t[2])
                    last_out_dim = t[1]
                self.glu_layers.append(glu_layer)

    def get_output(self, **inputs):
        fan_in = inputs["fan_in"]
        is_training = inputs["is_training"]
        if inputs.has_key("keep_prob"):
            keep_prob = inputs["keep_prob"]
        else:
            keep_prob = 1.
        l2_loss = 0

        for glu_layer, t in zip(self.glu_layers, self.paras):
            fan_out, l2_loss = glu_layer.get_output(fan_in=fan_in,
                                                    is_training=is_training,
                                                    keep_prob=keep_prob)
            fan_in = fan_out
            l2_loss += l2_loss

        return fan_out, l2_loss


#### block by YinanXu ####
class ProjectionLayer(BasicLayer):
    def __init__(self,
                 name,
                 in_dim,
                 dim,
                 initializer=None):
        self.initializer = initializer
        super(self.__class__, self).__init__(in_dim=in_dim,
                                             dim=dim,
                                             name=name)

    def set_variables(self):
        in_dim = self.in_dim
        dim = self.dim
        with tf.variable_scope(self.name, initializer=self.initializer):
            proj_W = tf.get_variable("projection_W",
                shape=[in_dim, dim])
            proj_b = tf.get_variable("projection_b",
                shape=[dim])

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]
        if "name" in inputs:
            name = inputs["name"]
        else:
            name = None

        with tf.variable_scope(self.name, reuse=True):
            proj_W = tf.get_variable("projection_W")
            proj_b = tf.get_variable("projection_b")
            logits = tf.add(tf.matmul(fan_in, proj_W), proj_b, name=name)

        return logits


#### block by YinanXu ####
class DifferenceProjectionLayer(BasicLayer):
    def __init__(self,
                 name,
                 in_dim,
                 dim,
                 initializer=None):
        self.initializer = initializer
        super(self.__class__, self).__init__(in_dim=in_dim,
                                             dim=dim,
                                             name=name)

    def set_variables(self):
        in_dim = self.in_dim
        dim = self.dim
        with tf.variable_scope(self.name, initializer=self.initializer):
            proj_W = tf.get_variable("product_W",
                shape=[in_dim, dim])
            diff_W = tf.get_variable("difference_W",
                shape=[in_dim, dim])
            bias = tf.get_variable("bias",
                shape=[dim])

    def build_layer(self, **inputs):
        prod_in = inputs["prod"]
        diff_in = inputs["diff"]

        with tf.variable_scope(self.name, reuse=True):
            proj_W = tf.get_variable("product_W")
            diff_W = tf.get_variable("difference_W")
            bias = tf.get_variable("bias")

            out = tf.nn.sigmoid(tf.matmul(prod_in, proj_W) + tf.matmul(diff_in, diff_W) + bias)

        return out


#### block by NickPAN ####
class HierarchicalSoftmax(BasicLayer):
    """
    Two-layer Hierarchical Softmax layer. Provides an approximate.
    Args:
        -fan_in: A tensor with shape [batch_size, in_dim].
        -target: A 1D tensor with shape [batch_size] or None.
        -batch_size: int, batch_size of fan_in.
        -total_outputs: int, how many outputs the hierarchical softmax is over.
        -per_class: int, how many outputs per top level class.
    Return:
        -h_softmax_out: tensor with shape [batch_size, 1] or
            [batch_size, total_outputs] or [batch_size, num_classes]
    """
    def __init__(self,
                 name,
                 batch_size,
                 total_outputs,
                 per_class,
                 in_dim,
                 initializer=None):
        self.batch_size = batch_size
        self.total_outputs = total_outputs
        self.per_class = per_class
        self.n_classes = int(np.ceil(self.total_outputs * 1. / self.per_class))
        self.n_outputs_actual = self.n_classes * self.per_class
        self.initializer = initializer

        assert self.n_outputs_actual >= self.total_outputs
        super(self.__class__, self).__init__(name=name, in_dim=in_dim)

    def set_variables(self):
        in_dim = self.in_dim
        total_outputs = self.total_outputs
        per_class = self.per_class
        n_classes = self.n_classes
        with tf.variable_scope(self.name, initializer=self.initializer):
            top_W = tf.get_variable("top_weights",
                shape=[in_dim, n_classes])
            top_b = tf.get_variable("top_bias",
                shape=[n_classes],
                initializer=constant_initializer(0))

            bottom_W = tf.get_variable("bottom_weights",
                shape=[n_classes, in_dim, per_class])
            bottom_b = tf.get_variable("bottom_bias",
                shape=[n_classes, per_class],
                initializer=constant_initializer(0))

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]
        if inputs.has_key("target"):
            target = inputs["target"]
        else:
            target = None
        batch_size = self.batch_size
        in_dim = self.in_dim
        total_outputs = self.total_outputs
        per_class = self.per_class
        n_classes = self.n_classes

        with tf.variable_scope(self.name, reuse=True):
            top_W = tf.get_variable("top_weights")
            top_b = tf.get_variable("top_bias")
            bottom_W = tf.get_variable("bottom_weights")
            bottom_b = tf.get_variable("bottom_bias")

            output_probs = h_softmax(fan_in, batch_size, total_outputs,
                n_classes, per_class, top_W, top_b,
                bottom_W, bottom_b, target=target)

        return output_probs


#### block by NickPAN ####
class TreeHierarchicalSoftmax(BasicLayer):
    """
    Full hierarchical softmax layer. A tree structure of labels should be
    provided.
    Args:
        -tree: A tree object of hierarchical labels.
    Return:
        A tensor with shape [batch_size, 1]
    """
    def __init__(self,
                 name,
                 tree,
                 batch_size,
                 in_dim,
                 initializer=None):
        self.tree = tree
        self.batch_size = batch_size
        self.initializer = initializer

        super(self.__class__, self).__init__(name=name, in_dim=in_dim)

    def set_variables(self):
        in_dim = self.in_dim
        batch_size = self.batch_size
        tree = self.tree
        num_nodes = tree.count_nodes(tree)  # all nodes incloud root
        max_classes = tree.find_max_classes(tree)  # max classes in nodes
        with tf.variable_scope(self.name, initializer=self.initializer):
            W = tf.get_variable("tree_W",
                shape=[num_nodes, in_dim, max_classes])

            b = tf.get_variable("tree_b",
                shape=[num_nodes, max_classes],
                initializer=constant_initializer(0))

    def build_layer(self, **inputs):
        fan_in = inputs["fan_in"]
        label = inputs["label"]
        slice_idx = inputs["slice_idx"]
        label_length = inputs["label_length"]

        with tf.variable_scope(self.name, reuse=True):
            # computation is samplewise
            fan_in_list = tf.unpack(fan_in)
            label_list = tf.unpack(label)
            slice_idx_list = tf.unpack(slice_idx)
            label_length_list = tf.unpack(label_length)
            W = tf.get_variable("tree_W")
            b = tf.get_variable("tree_b")

            batch_probs_list = []
            for _fan_in, _label, _slice, _label_length in itertools.izip(
                    fan_in_list, label_list,
                    slice_idx_list, label_length_list):
                prob = tree_softmax(_fan_in, _label,
                                    _slice, _label_length,
                                    W, b)
                batch_probs_list.append(prob)

        return tf.pack(batch_probs_list)
