import tensorflow as tf
import numpy as np

def convolve(x, n_filters, outsize):
    x = tf.reshape(x, [-1, 1, x.shape[1], 1])
    w_conv = tf.Variable(tf.random_normal([1, 1, 1, n_filters]))
    b_conv = tf.Variable(tf.random_normal([n_filters]))
    conv = tf.nn.convolution(input=x, filter=w_conv, padding="VALID", strides=[1, 1])
    conv = tf.nn.relu(tf.add(conv, b_conv))
    pool = tf.nn.pool(
        input=conv,
        window_shape=[1, 1],
        pooling_type="MAX",
        padding="SAME",
        strides=[1, 1],
    )
    data_len = pool.shape[2]
    filter_depth = pool.shape[3]
    size = int(data_len * filter_depth)

    pool = tf.reshape(pool, [-1, size])
    pool = tf.convert_to_tensor(pool, dtype=tf.float32)

    W = tf.random_normal(shape=[size, outsize], mean=0.2, stddev=0.2)
    b = tf.random_normal(shape=[outsize], mean=0.2, stddev=0.2)

    xconv = tf.nn.relu(tf.add(tf.matmul(pool, W), b))
    #   xconv = tf.nn.tanh(tf.add(tf.matmul(pool, W), b))

    return xconv


def weights(input_vector, in_nodes, out_nodes):
    """
    Set up node weights and biases.
    input_vector: tf.placeholder for input vector
    in_nodes: int - number of nodes into layer
    out_nodes: int - number of nodes out of layer
    """
    W = tf.random_normal(shape=[in_nodes, out_nodes], mean=0.2, stddev=0.2)
    b = tf.random_normal(shape=[out_nodes], mean=0.2, stddev=0.2)
    return build_layer(input_vector, tf.Variable(W), tf.Variable(b))


def get_batch(vectors, labels):
    """
    Create a batch of 1000 datapoints 
    on which to train from the complete
    set of training vectors and labels.
    """
    indices = np.arange(len(vectors))
    np.random.shuffle(indices)
    chosen = indices[:1000]
    return vectors[chosen], labels[chosen]


def build_layer(x, W, b):
    """
    Multiply an input vector x (tf.placeholder) with
    the node weights (W) and add node bias (b).
    """
    return tf.add(tf.matmul(x, W), b)


def run_net(
    train,
    test,
    Nlayers=3,
    N_nodes=[10, 5, 1],
    training_iterations=1e3,
    run_name="",
    show_comparison=False,
    nfilters=27,
    convolution_outsize=10,
    forward_hops_only=False,
):
    """
    Run the Neural Net. Used mainly when applying the net in
    a jupyter notbooke to prevent duplicating code.
    """

    train_features = train.drop(["chromophoreA", "chromophoreB", "TI"], axis=1)
    test_features = test.drop(["chromophoreA", "chromophoreB", "TI"], axis=1)
    train_labels = train[["TI"]]
    test_labels = test[["TI"]]
    test_answers = test_labels

    train_features, test_features, train_labels, test_labels = (
        train_features.values,
        test_features.values,
        train_labels.values,
        test_labels.values,
    )

    assert Nlayers == len(N_nodes)

    x = tf.placeholder(tf.float32, shape=[None, len(train_features[0])])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    layer_dict = {}

    xconv = convolve(x, nfilters, N_nodes[0])

    for N in range(Nlayers):
        if N == 0:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(
                weights(x, N_nodes[0], N_nodes[N])
            )
        elif N + 1 == Nlayers:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(
                weights(
                    layer_dict["layer_{}".format(N - 1)], N_nodes[N - 1], N_nodes[N]
                )
            )
        else:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(
                weights(
                    layer_dict["layer_{}".format(N - 1)], N_nodes[N - 1], N_nodes[N]
                )
            )

    y_out = layer_dict["layer_{}".format(Nlayers - 1)]

    cost = tf.losses.huber_loss(labels=y_, predictions=y_out)

    optimizer = tf.train.AdamOptimizer(0.01)
    training_step = optimizer.minimize(cost)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    training_error = []

    for _ in range(int(training_iterations)):
        batch_vectors, batch_answers = get_batch(train_features, train_labels)
        session.run(training_step, feed_dict={x: batch_vectors, y_: batch_answers})
        if _ % (int(training_iterations) // 10) == 0:

            train_error = session.run(
                cost, feed_dict={x: train_features, y_: train_labels}
            )
            training_error.append([_, train_error])
            print(train_error)

    pred_y = session.run(y_out, feed_dict={x: test_features})

    return pred_y
