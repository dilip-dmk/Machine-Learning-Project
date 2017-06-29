from __future__ import absolute_import
from __future__ import division


import tensorflow as tf

#### block by NickPAN ####
def h_softmax(x, batch_size, n_outputs, n_classes,
              n_outputs_per_class, W1, b1, W2, b2, target=None):
    """
    A tensorflow version of theano's two-level hierarchical softmax.
    Args:
        -x: tensor of shape (batch_size, number of features)
            the minibatch input of the two-layer hierarchical softmax.
        -batch_size: int, the size of the minibatch input x.
        -n_outputs: int, the number of outputs.
        -n_classes: int, the number of classes of the two-layer hierarchical
            softmax. It corresponds to the number of outputs of the first
            softmax.
        -n_outputs_per_class: int, the number of outputs per class.
        -W1: tensor of shape (number of features of the input x, n_classes)
            the weight matrix of the first softmax, which maps the input x to
            the probabilities of the classes.
        -b1: tensor of shape (n_classes,) the bias vector of the first
            softmax layer.
        -W2: tensor of shape (n_classes, number of features of the input x,
            n_outputs_per_class) the weight matrix of the second softmax,
            which maps the input x to the probabilities of the outputs.
        -b2: tensor of shape (n_classes, n_outputs_per_class), the bias vector
            of the second softmax layer.
        -target: tensor of shape either (batch_size,) or (batch_size, 1),
            (optional, default None), contains the indices of the targets for
            the minibatch input x. For each input, the function computes the
            output for its corresponding target. If target is None, then all
            the outputs are computed for each input.
    Returns:
            tensor of shape (`batch_size`, `n_outputs`) or (`batch_size`, 1)
            Output tensor of the two-layer hierarchical softmax for input `x`.
            Depending on argument `target`, it can have two different shapes.
            If `target` is not specified (`None`), then all the outputs are
            computed and the returned tensor has shape (`batch_size`, `n_outputs`).
            Otherwise, when `target` is specified, only the corresponding outputs
            are computed and the returned tensor has thus shape (`batch_size`, 1).
    """
    # First softmax that computes the probabilities of belonging to each class
    class_probs = tf.nn.softmax(tf.matmul(x, W1) + b1)  # [batch_size, n_classes]

    if target is None:
        # Computes the probabilites of all the outputs
        # Second softmax that computes the output probabilities
        activations = tf.einsum("bi,kin->bkn", x, W2) + b2
        output_probs = tf.nn.softmax(tf.reshape(activations,
            shape=[-1, n_outputs_per_class]))
        output_probs = tf.reshape(output_probs,
            shape=[batch_size, n_classes, -1])   # [batch_size, n_classes, n_outputs_per_class]
        output_probs = tf.expand_dims(class_probs, 2) * output_probs
        output_probs = tf.reshape(output_probs, shape=[batch_size, -1])
        # output_probs.shape[1] is n_classes * n_outputs_per_class, which might
        # be greater than n_outputs, so we ignore the potential irrelevant
        # outputs with the next line:
        output_probs = output_probs[:, :n_outputs]

    else:
        # Computes the probabilities of the outputs specified by the targets
        target = tf.cast(tf.reshape(target, shape=[-1]), tf.int32)

        # Classes to which belong each target
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_outputs_per_class

        # Second softmax that computes the output probabilities
        _W2 = tf.gather(W2, target_classes)  # [batch_size, in_dim, n_outputs_per_class]
        _b2 = tf.gather(b2, target_classes)  # [batch_size, n_outputs_per_class]
        _activations = tf.batch_matmul(tf.expand_dims(x, 1), _W2)  # [batch_size, 1, n_outputs_per_class]
        activations = tf.squeeze(_activations, [1]) + _b2  # [batch_size, n_outputs_per_class]

        output_probs = tf.nn.softmax(activations)
        target_class_probs = tf.reduce_sum(
            class_probs * tf.one_hot(target_classes, depth=n_classes), axis=1,
            keep_dims=True)  # [batch_size, 1] for probs be in the target_classes
        output_probs = tf.reduce_sum(
            output_probs * tf.one_hot(target_outputs_in_class,
            depth=n_outputs_per_class), axis=1, keep_dims=True)  # [batch_size, 1] for probs be the target_outputs_in_class
        output_probs = target_class_probs * output_probs

    return output_probs


#### block by NickPAN ####
def tree_softmax(x, label, slice_idx, label_length, W, b):
    """
    Tree hierarchical softmax.
    Args:
        -x: 1D tensor with [hidden_size].
        -label: 1D tensor with [label_max_length].
        -slice_idx: 3D tensor with [label_max_length, 2, 3].
        -label_length: 0D tensor contains true length of the label.
        -W, 3D tensor with [num_nodes, hidden_size, max_classes].
        -b, 2D tensor with [num_nodes, max_classes].
    Return:
        Probability of x in class labeled by label

    Example:
        Label is: [4 1 3 0 -1]
        Label_length is : [4]
        Slice is: [[[   0    0    0]
                    [   1 1200   11]]

                   [[ 935    0    0]
                    [   1 1200    5]]

                   [[ 954    0    0]
                    [   1 1200    6]]

                   [[ 980    0    0]
                    [   1 1200    5]]

                   [[ -1    -1    -1]
                    [ -1    -1    -1]]]
    """
    _x = tf.expand_dims(x, 0)  # [hidden_size] to [1, hidden_size]

    # defind the cond for while_loop
    def cond(i, prob, label, label_length, slice_idx, W, b, x):
        return tf.less(i, label_length)

    # defind the main computation
    def loop(i, prob, label, label_length, slice_idx, W, b, x):
        _label = tf.gather(label, i)
        _slice_idx = tf.gather(slice_idx, i)

        # get rid of second row of _slice to get slice index for b
        # _slice_b_idx_t = tf.gather(tf.transpose(_slice_idx), [0,2])
        _slice_b_idx = tf.gather_nd(_slice_idx,
                                    indices=[[[0, 0], [0, 2]],
                                             [[1, 0], [1, 2]]]
                        )

        # the last element of second row indicates the number of classes
        num_classes = tf.gather_nd(_slice_b_idx, [1, 1])

        _sliced_W = tf.slice(W, begin=tf.gather(_slice_idx, 0),
                             size=tf.gather(_slice_idx, 1))
        _sliced_b = tf.slice(b, begin=tf.gather(_slice_b_idx, 0),
                             size=tf.gather(_slice_b_idx, 1))
        sliced_W = tf.squeeze(_sliced_W, axis=0)
        sliced_b = tf.squeeze(_sliced_b, axis=0)

        logits = tf.matmul(x, sliced_W) + sliced_b
        probs = tf.nn.softmax(logits)

        # get one-hot label and compute prob
        prob = prob * tf.reduce_sum(tf.one_hot(_label, num_classes) * probs)

        return [tf.add(i, 1), prob, label, label_length, slice_idx, W, b, x]


    i = tf.constant(0, dtype=tf.int32)
    prob = tf.constant(1.0, dtype=tf.float32)
    loop_list = tf.while_loop(cond,
                              loop,
                              loop_vars=[i, prob, label,
                                         label_length,
                                         slice_idx, W, b, _x])

    return loop_list[1]
