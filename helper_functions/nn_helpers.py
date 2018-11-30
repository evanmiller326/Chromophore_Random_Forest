import tensorflow as tf
#import ANN

def run_net(train, 
            test, 
        Nlayers = 3, 
        N_nodes= [10, 5, 1], 
        training_iterations = 1e3, 
        run_name = "", 
        show_comparison = False, 
        nfilters = 27,
        convolution_outsize = 10,
        forward_hops_only = False):

    train_features = train.drop(['chromophoreA', 'chromophoreB', 'TI'], axis=1)
    test_features = test.drop(['chromophoreA', 'chromophoreB', 'TI'], axis=1)
    train_labels = train[['TI']]
    test_labels = test[['TI']]
    test_answers = test_labels
    
    train_features, test_features, train_labels, test_labels = train_features.values, test_features.values, train_labels.values, test_labels.values

    assert Nlayers == len(N_nodes)

    x = tf.placeholder(tf.float32, shape = [None, len(train_features[0])])
    y_ = tf.placeholder(tf.float32, shape = [None, 1])

    layer_dict = {}

    xconv = ANN.convolve(x, nfilters, convolution_outsize)

    for N in range(Nlayers):
        if N == 0:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(ANN.weights(x, convolution_outsize, N_nodes[N]))
        elif N + 1 == Nlayers:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(ANN.weights(layer_dict["layer_{}".format(N-1)], N_nodes[N-1], N_nodes[N]))
        else:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(ANN.weights(layer_dict["layer_{}".format(N-1)], N_nodes[N-1], N_nodes[N]))

    y_out = layer_dict["layer_{}".format(Nlayers-1)]

    cost = tf.losses.huber_loss(labels = y_, predictions = y_out)

    optimizer = tf.train.AdamOptimizer(0.01)
    training_step = optimizer.minimize(cost)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    training_error = []

    for _ in range(int(training_iterations)):
        batch_vectors, batch_answers = ANN.get_batch(train_features, train_labels)
        session.run(training_step,feed_dict={x:batch_vectors,y_:batch_answers})
        if _ % (int(training_iterations)//10) == 0:

            train_error = session.run(cost,feed_dict={x:train_features,y_:train_labels})
            training_error.append([_, train_error])
            print(train_error)

    pred_y = session.run(y_out, feed_dict={x: test_features})

    mae = np.mean(abs(pred_y-test_labels))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        test_labels.flatten(), pred_y.flatten()
    )

    return pred_y
