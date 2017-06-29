from __future__ import absolute_import
from __future__ import division

import collections

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops

from layers import RNNBatchnromLayer

#### block by NickPAN ####
def _bn_linear(is_training,
               args, output_size,
               bias, bias_start=0.0,
               scope=None):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        -is_training: 0D bool tensor, control the update of running mean and
            output of batchnorme.
        -args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        -output_size: int, second dimension of W[i].
        -bias: boolean, whether to add a bias term or not.
        -bias_start: starting value to initialize the bias; 0 by default.
        -scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(BN(args[i] * W[i])), where W[i]s are newly created matrices.
    Raises:
        -ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation
    out_list = []
    with vs.variable_scope(scope or "Linear"):
        for i, shape in enumerate(shapes):
            mx_name = "Matrix_" + str(i)
            matrix = tf.get_variable(
                mx_name, shape=[shape[1], output_size], dtype=dtype)
            res = tf.matmul(args[i], matrix)

            # Batchnorm
            # if use bias or bias is 0, then batchnorm should uncentered
            if bias or bias_start == 0.0:
                is_center = False
            else:
                is_center = True
            bn_name = "batchnorm_" + str(i)
            rnn_bn_layer = RNNBatchnromLayer(bn_name, center=is_center)
            bn_res = rnn_bn_layer.get_output(
                fan_in=res, is_training=is_training)

            # Bias
            if not bias:
                out_list.append(bn_res)
            else:
                bi_name = "Bias_" + str(i)
                bias_term = tf.get_variable(
                    bi_name, shape=[output_size],
                    dtype=dtype,
                    initializer=tf.constant_initializer(
                        bias_start, dtype=dtype))
                out_list.append(bn_res + bias_term)

    return tf.add_n(out_list)


#### block by NickPAN ####
class BN_GRUCell(GRUCell):
    """
    Batchnorm of Gated recurrent unit.
    http://arxiv.org/abs/1406.1078
    https://arxiv.org/abs/1603.09025
    Args:
        -is_training: 0D boolean tensor.
    """
    def __init__(self, num_units,
                 is_training,
                 input_size=None,
                 activation=tf.tanh):
        self.is_training = is_training
        super(self.__class__, self).__init__(num_units=num_units,
                                             input_size=input_size,
                                             activation=activation)


    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates_r"):
                r = _bn_linear(
                    self.is_training, [inputs, state],
                    self._num_units, True, 1.0)
            with tf.variable_scope("Gates_u"):
                u = _bn_linear(
                    self.is_training, [inputs, state],
                    self._num_units, True, 1.0)

            r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = self._activation(_bn_linear(self.is_training,
                                                [inputs, r * state],
                                                self._num_units,
                                                True)
                                    )

            new_h = u * state + (1 - u) * c

        return new_h, new_h


#### block by NickPAN ####
#### Hierarchical Multiscale RNN Cells ####
# Note: Requires Tensorflow 0.11 or lower. Breaks in Tensorflow 0.12.
def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    If x is >= 0.5, binaryRound(x) will be 1 and the gradient will be pass-through,
    otherwise, binaryRound(x) will be 0 and the gradient will be 0.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        # override "Floor" because tf.round uses tf.floor
        with g.gradient_override_map({"Round": "BinaryRound"}):
            return tf.round(x, name=name)

@ops.RegisterGradient("BinaryRound")
def _binaryRound(op, grad):
    """Straight through estimator for the binaryRound op (identity if 1, else 0)."""
    x = op.outputs[0]
    return x * grad

def bernoulliSample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
    using the straight through estimator for the gradient.

    E.g.,:
    if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
    and the gradient will be pass-through (1) wih probability 0.6, and 0 otherwise.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity","Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

@ops.RegisterGradient("BernoulliSample_ST")
def bernoulliSample_ST(op, grad):
    """Straight through estimator for the bernoulliSample op (identity if 1, else 0)."""
    sub = op.outputs[0] # x - tf.random_uniform...
    res = sub.consumers()[0].outputs[0] # tf.ceil(sub)
    return [res * grad, tf.zeros(tf.shape(op.inputs[1]))]

def passThroughSigmoid(x, slope=1):
    """Sigmoid that uses identity function as its gradient"""
    g = tf.get_default_graph()
    with ops.name_scope("PassThroughSigmoid") as name:
        with g.gradient_override_map({"Sigmoid": "Identity"}):
            return tf.sigmoid(x, name=name)

def binaryStochastic_ST(x, slope_tensor=None, pass_through=True, stochastic=True):
    """
    Sigmoid followed by either a random sample from a bernoulli distribution according
    to the result (binary stochastic neuron) (default), or a sigmoid followed by a binary
    step function (if stochastic == False). Uses the straight through estimator.
    See https://arxiv.org/abs/1308.3432.

    Arguments:
    * x: the pre-activation / logit tensor
    * slope_tensor: if passThrough==False, slope adjusts the slope of the sigmoid function
        for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
    * pass_through: if True (default), gradient of the entire function is 1 or 0;
        if False, gradient of 1 is scaled by the gradient of the sigmoid (required if
        Slope Annealing Trick is used)
    * stochastic: binary stochastic neuron if True (default), or step function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    #TODO hard sigmoid:
    # z_tilda = tf.maximum(0, tf.minimum(1, (slope * z_t_logit) / 2))
    if pass_through:
        p = passThroughSigmoid(x)  # TODO hard sigmoid? pass though it typically used when we don't do slope annealing
    else:
        p = tf.sigmoid(slope_tensor*x) # TODO hard sigmoid

    if stochastic:
        return bernoulliSample(p)
    else:
        return binaryRound(p)


def binary_wrapper(pre_activations_tensor,
                   stochastic_tensor=tf.constant(True),
                   pass_through=True,
                   slope_tensor=tf.constant(1.0)):
    """
    Turns a layer of pre-activations (logits) into a layer of binary stochastic neurons

    Keyword arguments:
    *stochastic_tensor: a boolean tensor indicating whether to sample from a bernoulli
        distribution (True, default) or use a step_function (e.g., for inference)
    *pass_through: for ST only - boolean as to whether to substitute identity derivative on the
        backprop (True, default), or whether to use the derivative of the sigmoid
    *slope_tensor: for ST only - tensor specifying the slope for purposes of slope annealing
        trick
    """
    if pass_through:
        return tf.cond(stochastic_tensor,
                       lambda: binaryStochastic_ST(pre_activations_tensor),
                       lambda: binaryStochastic_ST(pre_activations_tensor,
                                                   stochastic=False))
    else:
        return tf.cond(stochastic_tensor,
                       lambda: binaryStochastic_ST(pre_activations_tensor,
                                                   slope_tensor = slope_tensor,
                                                   pass_through=False),
                       lambda: binaryStochastic_ST(pre_activations_tensor,
                                                   slope_tensor = slope_tensor,
                                                   pass_through=False,
                                                   stochastic=False))
# TODO end s2st code


_HmGruStateTuple = collections.namedtuple("GRUStateTuple", ("h", "z"))

class HmGruStateTuple(_HmGruStateTuple):
    """Tuple used by HmGru Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(h, z)`, in that order.
    """
    __slots__ = ()

    @property
    def dtype(self):
      (h, z) = self
      return h.dtype


class HmGruCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units):
      # self._num_units determines the size of h.
      self._num_units = num_units

    @property
    def state_size(self):
      return HmGruStateTuple(self._num_units, 1)

    @property
    def output_size(self):
      return self._num_units

    # TODO make the type changeable instead of defaulting to float32?

    # All RNNs return (output, new_state). For this type of GRU, its 'output' is still its h vector,
    # and it's cell state is a h,z tuple.
    # 'inputs' are a tuple of h_bottom, z_bottom, and h_top_prev.
    # 'state' is a h,z HmGruTuple
    def __call__(self, inputs, state, scope=None):
        # vars from different layers.
        h_bottom, z_bottom, h_top_prev = inputs
        # vars from the previous time step on the same layer
        h_prev, z_prev = state

        # I'm calling the the 'z gate' in GRU the 'o gate', since z means something different in HM-LSTM.
        # Not including the candidate hidden state (c_tilda, or g as I call it, since it needs to be
        # multiplied by r first.
        # Need enough rows in the shared matrix for r, o, z_stochastic_tilda
        num_rows = 2 * self._num_units + 1

        # scope: optional name for the variable scope, defaults to "HmGruCell"
        with vs.variable_scope(scope or type(self).__name__):
            # Matrix U_l^l
            U_curr = vs.get_variable("U_curr",
                                     [h_prev.get_shape()[1], num_rows],
                                     dtype=tf.float32)
            # Matrix U_{l+1}^l
            U_top = vs.get_variable("U_top",
                                    [h_bottom.get_shape()[1], num_rows],
                                    dtype=tf.float32)
            # Matrix W_{l-1}^l
            W_bottom = vs.get_variable("W_bottom",
                                       [h_bottom.get_shape()[1], num_rows],
                                       dtype=tf.float32)
            # b_l
            bias = vs.get_variable("bias", [num_rows], dtype=tf.float32)

            s_curr = tf.matmul(h_prev, U_curr)
            s_top = z_prev * tf.matmul(h_top_prev, U_top)
            s_bottom = z_bottom * tf.matmul(h_bottom, W_bottom)
            gate_logits = s_curr + s_top + s_bottom + bias

            r_logits = tf.slice(gate_logits, [0, 0], [-1, self._num_units])
            o_logits = tf.slice(gate_logits,
                                [0, self._num_units],
                                [-1, self._num_units])
            z_t_logit = tf.slice(gate_logits, [0, 2*self._num_units], [-1, 1])

            r = tf.sigmoid(r_logits)
            o = tf.sigmoid(o_logits)
            # This is the stochastic neuron
            z_new = binary_wrapper(z_t_logit,
                                   pass_through=False, # TODO make this true if you do slope annealing
                                   stochastic_tensor=tf.constant(True), # TODO make this false if you do slope annealing
                                   slope_tensor=None) # TODO set this if you do slope annealing

            # Now calculate the candidate gate (c_tilda aka g)
            # Matrix U_l^l (for just g)
            U_g_curr = vs.get_variable("U_g_curr",
                                       [h_prev.get_shape()[1],
                                       self._num_units],
                                       dtype=tf.float32)
            # Matrix U_{l+1}^l (for just g)
            U_g_top = vs.get_variable("U_g_top",
                                      [h_bottom.get_shape()[1],
                                      self._num_units],
                                      dtype=tf.float32)
            # Matrix W_{l-1}^l (for just g)
            W_g_bottom = vs.get_variable("W_g_bottom",
                                         [h_bottom.get_shape()[1],
                                         self._num_units],
                                         dtype=tf.float32)
            # b_l (for just g)
            bias_g = vs.get_variable("bias_g", [self._num_units], dtype=tf.float32)
            s_g_curr = tf.matmul(r * h_prev, U_g_curr)
            s_g_top = z_prev * tf.matmul(r * h_top_prev, U_g_top)
            s_g_bottom = z_bottom * tf.matmul(r * h_bottom, W_g_bottom)
            g_logits = s_g_curr + s_g_top + s_g_bottom + bias_g
            g = tf.tanh(g_logits)

            z_zero_mask = tf.equal(z_prev, tf.zeros_like(z_prev))
            copy_mask = tf.to_float(tf.logical_and(z_zero_mask,
                                                   tf.equal(z_bottom,
                                                            tf.zeros_like(z_bottom)
                                                   )
                                    )
                        )
            update_mask = tf.to_float(tf.logical_and(z_zero_mask, tf.cast(z_bottom, tf.bool)))
            flush_mask = z_prev

           # TODO put this behind a test flag
           # tf.assert_equal(tf.reduce_sum(copy_mask + update_mask + flush_mask),
           #                tf.reduce_sum(tf.ones_like(flush_mask))) # TODO
            h_flush = o * g
            h_update = (tf.ones_like(o) - o) * h_prev + h_flush
            h_new = copy_mask * h_prev + update_mask * h_update + flush_mask * h_flush

            return h_new, HmGruStateTuple(h_new, z_new)



_HmLstmStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "z"))


#### block by NickPAN ####
class HmLstmStateTuple(_HmLstmStateTuple):
    """Tuple used by HmLstm Cells for `state_size`, `zero_state`, and output state.
    Stores three elements: `(c, h, z)`, in that order.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, z) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


#### block by NickPAN ####
class HmLstmCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units):
        # self._num_units determines the size of c and h.
        self._num_units = num_units

    @property
    def state_size(self):
        return HmLstmStateTuple(self._num_units, self._num_units, 1)

    @property
    def output_size(self):
        return self._num_units

    # TODO make the type changeable instead of defaulting to float32?

    # All RNNs return (output, new_state). For this type of LSTM, its 'output' is still its h vector,
    # and it's cell state is a c,h,z tuple.
    # 'inputs' are a tuple of h_bottom, z_bottom, and h_top_prev.
    # 'state' is a c,h,z HmLstmTuple
    def __call__(self, inputs, state, scope=None):
        # vars from different layers.
        h_bottom, z_bottom, h_top_prev = inputs
        # vars from the previous time step on the same layer
        c_prev, h_prev, z_prev = state

        # Need enough rows in the shared matrix for f, i, o, g, z_stochastic_tilda
        num_rows = 4 * self._num_units + 1

        # scope: optional name for the variable scope, defaults to "HmLstmCell"
        with vs.variable_scope(scope or type(self).__name__):  # "HmLstmCell"
            # Matrix U_l^l
            U_curr = vs.get_variable("U_curr",
                                     [h_prev.get_shape()[1], num_rows],
                                     dtype=tf.float32)
            # Matrix U_{l+1}^l
            # TODO This imples that the U matrix there has the same dimensionality as the
            # one used in equation 5. but that would only be true if you forced the h vectors
            # on the above layer to be equal in size to the ones below them. Is that a real restriction?
            # Or am I misunderstanding?
            U_top = vs.get_variable("U_top",
                                    [h_bottom.get_shape()[1], num_rows],
                                    dtype=tf.float32)
            # Matrix W_{l-1}^l
            W_bottom = vs.get_variable("W_bottom",
                                       [h_bottom.get_shape()[1], num_rows],
                                       dtype=tf.float32)
            # b_l
            bias = vs.get_variable("bias", [num_rows], dtype=tf.float32)

            s_curr = tf.matmul(h_prev, U_curr)
            s_top = z_prev * tf.matmul(h_top_prev, U_top)
            s_bottom = z_bottom * tf.matmul(h_bottom, W_bottom)
            gate_logits = s_curr + s_top + s_bottom + bias

            f_logits = tf.slice(gate_logits, [0, 0], [-1, self._num_units])
            i_logits = tf.slice(gate_logits, [0, self._num_units], [-1, self._num_units])
            o_logits = tf.slice(gate_logits, [0, 2*self._num_units], [-1, self._num_units])
            g_logits = tf.slice(gate_logits, [0, 3*self._num_units], [-1, self._num_units])
            z_t_logit = tf.slice(gate_logits, [0, 4*self._num_units], [-1, 1])

            f = tf.sigmoid(f_logits)
            i = tf.sigmoid(i_logits)
            o = tf.sigmoid(o_logits)
            g = tf.tanh(g_logits)

            # This is the stochastic neuron
            z_new = binary_wrapper(z_t_logit,
                                   pass_through=False, # TODO make this true if you do slope annealing
                                   stochastic_tensor=tf.constant(True), # TODO make this false if you do slope annealing
                                   slope_tensor=None) # TODO set this if you do slope annealing

            z_zero_mask = tf.equal(z_prev, tf.zeros_like(z_prev))
            copy_mask = tf.to_float(
                tf.logical_and(z_zero_mask,
                               tf.equal(z_bottom, tf.zeros_like(z_bottom))
                )
            )
            update_mask = tf.to_float(tf.logical_and(z_zero_mask, tf.cast(z_bottom, tf.bool)))
            flush_mask = z_prev

            # TODO put this behind a test flag
            # tf.assert_equal(tf.reduce_sum(copy_mask + update_mask + flush_mask),
            # tf.reduce_sum(tf.ones_like(flush_mask))) # TODO

            c_flush = i * g
            c_update = f * c_prev + c_flush
            c_new = copy_mask * c_prev + update_mask * c_update + flush_mask * c_flush

            h_flush = o * tf.tanh(c_flush)
            h_update = o * tf.tanh(c_update)
            h_new = copy_mask * h_prev + update_mask * h_update + flush_mask * h_flush

            state_new = HmLstmStateTuple(c_new, h_new, z_new)
            return h_new, state_new


#### block by NickPAN ####
# The output for this is a list of h_vectors, one for each cell.
class MultiHmRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cells, output_embedding_size=None):
        """Create a RNN cell composed sequentially of a number of HmRNNCells.
        Args:
          cells: list of HmRNNCells that will be composed in this order.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiHmRNNCell.")
        self._cells = cells
        self._output_embedding_size = output_embedding_size

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._output_embedding_size

    # 'inputs' should be a batch of word vectors
    # 'state' should be a list of HM cell state tuples of the same length as self._cells
    # 'slope' is a scalar tensor for slope annealing.
    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        assert len(state) == len(self._cells)
        with vs.variable_scope(scope or type(self).__name__):  # "MultiHmRNNCell"
            if len(self._cells) > 1:
                h_prev_top = state[1].h
            else:
                h_prev_top = tf.zeros(state[0].h.get_shape())
            # h_bottom, z_bottom, h_prev_top
            current_input = inputs, tf.ones([inputs.get_shape()[0], 1]), h_prev_top
            new_h_list = []
            new_states = []
            # Go through each cell in the different layers, going bottom to top
            for i, cell in enumerate(self._cells):
                with vs.variable_scope("Cell%d" % i):
                    new_h, new_state = cell(current_input, state[i]) # state[i] = c_prev, h_prev, z_prev
                    #assert new_h == new_state.h # This isn't true if dropout is enabled
                    # Set up the inputs for the next cell.
                    if i < len(self._cells) - 2:
                        # Next cell is not the top one.
                        h_prev_top = state[i+2].h
                    else:
                        # The next cell is the top one, so give it zeros for its h_prev_top input.
                        h_prev_top = tf.zeros(state[i].h.get_shape())
                    current_input = new_state.h, new_state.z, h_prev_top  # h_bottom, z_bottom, h_prev_top
                    new_h_list.append(new_h)
                    new_states.append(new_state)
            # Output layer
            # may not need this
            # with vs.variable_scope("Output"):
            #     concat_new_h = tf.concat(0, new_h_list)
            #     output_logits = []
            #     for i in range(len(new_h_list)):
            #         # w^l
            #         gating_unit_weight = vs.get_variable("w{}".format(i), concat_new_h.get_shape(), dtype=tf.float32)
            #         # g_t^l
            #         gating_unit = tf.sigmoid(tf.reduce_sum(gating_unit_weight * concat_new_h))
            #         # W_l^e
            #         output_embedding_matrix = vs.get_variable("W{}".format(i),
            #                                                 [self._output_embedding_size, new_h_list[i].get_shape()[1]], dtype=tf.float32)
            #         output_logit = gating_unit * tf.matmul(new_h_list[i], output_embedding_matrix)
            #         output_logits.append(output_logit)
            #     output_h = tf.nn.relu(tf.add_n(output_logits))
            return new_h, tuple(new_states)
