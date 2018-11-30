import tensorflow as tf
import numpy as np
import sqlite3

import sklearn
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from scipy import stats

import matplotlib.pyplot as plt
import ml_helpers as mlh

def convolve(x, n_filters, outsize):
    x = tf.reshape(x, [-1, 1, x.shape[1], 1])
    w_conv = tf.Variable(tf.random_normal([1, 1, 1, n_filters]))
    b_conv = tf.Variable(tf.random_normal([n_filters]))
    conv = tf.nn.convolution(input = x, filter = w_conv, padding = "VALID", strides = [1, 1])
    conv = tf.nn.relu(tf.add(conv, b_conv))
    pool = tf.nn.pool(
            input=conv,
            window_shape= [1, 1],
            pooling_type = "MAX",
            padding = "SAME",
            strides = [1,1])
    data_len = pool.shape[2]
    filter_depth = pool.shape[3]
    size = int(data_len*filter_depth)

    pool = tf.reshape(pool, [-1, size])
    pool = tf.convert_to_tensor(pool, dtype=tf.float32)

    W = tf.random_normal(shape=[size, outsize], mean=0.2, stddev=0.2)
    b = tf.random_normal(shape=[outsize], mean = 0.2, stddev = 0.2)

    xconv = tf.nn.relu(tf.add(tf.matmul(pool, W), b))
 #   xconv = tf.nn.tanh(tf.add(tf.matmul(pool, W), b))

    return xconv

def weights(input_vector, in_nodes, out_nodes):
    W = tf.random_normal(shape=[in_nodes, out_nodes], mean=0.2, stddev=0.2)
    b = tf.random_normal(shape=[out_nodes], mean = 0.2, stddev = 0.2)
    #W = tf.random_normal(shape=[in_nodes, out_nodes], stddev=0.1)
    #b = tf.random_normal(shape=[out_nodes], stddev = 0.1)
    return build_layer(input_vector, tf.Variable(W), tf.Variable(b))

def get_batch(vectors, labels):
    indices = np.arange(len(vectors))
    np.random.shuffle(indices)
    chosen = indices[:1000]
    return vectors[chosen], labels[chosen]

def build_layer(x, W, b):
    return tf.add(tf.matmul(x,W),b)

def plot_comparison(actual, predicted, rmse, rsquared):
    plt.scatter(actual, predicted, zorder=0, alpha = 0.5, s=12)
    plt.plot(np.linspace(0, np.amax(actual), 10), np.linspace(0, np.amax(actual), 10), c='k', zorder = 10, label="MAE={:.3f},\nR$^2$={:.2f}".format(rmse, rsquared))
    #plt.title("ANN-RMSE-{:.5f}".format(rmse))
    plt.xticks([0.0, 0.5, 1.0, 1.5])
    plt.yticks([0.0, 0.5, 1.0, 1.5])
    plt.xlabel("Actual (eV)")
    plt.ylabel("Predicted (eV)")
    plt.legend(loc = 'upper left')
    plt.savefig("Ann_comp.png")

#Commented out to test if function is unused.
#def parallel_sort(a, b):
#    temp_a = np.copy(np.array(a))
#    temp_b = np.copy(np.array(b))
#    return zip(*sorted(zip(temp_a, temp_b)))

def find_largest_deviations(chromo_IDs, pred_y, actual_y):
    differences = np.array(actual_y) - np.array(pred_y)
    dictionary = {}
    error_dictionary = {diff[0]: chromo_IDs[index] for index, diff in enumerate(list(differences))}
    _, sorted_predictions = zip(*sorted(zip(np.copy(differences), pred_y)))
    _, sorted_actual = zip(*sorted(zip(np.copy(differences), actual_y)))
    sorted_keys = sorted(error_dictionary.keys())
    print("\nLargest overestimations =")
    for i in range(10):
        print("Chromos =", error_dictionary[sorted_keys[i]], "difference =", sorted_keys[i], ", predicted =", sorted_predictions[i], "actual =", sorted_actual[i])
    print("\nLargest underestimations =")
    for i in range(1, 11):
        print("Chromos =", error_dictionary[sorted_keys[-i]], "difference =", sorted_keys[-i], ", predicted =", sorted_predictions[-i], "actual =", sorted_actual[-i])
    print("\nLargest underestimations with predicted TIs < 0.01 eV (`the shelf') =")
    n = 0
    for i in range(len(sorted_keys)):
        if sorted_predictions[-i] < 0.01:
            print("Chromos =", error_dictionary[sorted_keys[-i]], "difference =", sorted_keys[-i], ", predicted =", sorted_predictions[-i], "actual =", sorted_actual[-i])
            n += 1
        if n == 10:
            break
    return error_dictionary

def plot_error_hist(error_dictionary):
    plt.figure()
    plt.hist(list(error_dictionary.keys()), bins=100)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.xlim([-0.5, 0.5])
    plt.savefig("./error_histogram_all.png")

def run_net(database,
        training,
        validation,
        absolute,
        skip,
        yval,
        Nlayers = 1, 
        N_nodes= [1], 
        training_iterations = 5e4, 
        run_name = "", 
        show_comparison = False, 
        nfilters = 27,
        convolution_outsize = 10,
        forward_hops_only = False):

    chromophore_ID_cols = ["chromophoreA", "chromophoreB"]
    for chromophore_ID_col in chromophore_ID_cols:
        if chromophore_ID_col not in skip:
            skip.append(chromophore_ID_col)
    # Also don't want to train on the answer!
    if yval not in skip:
        skip.append(yval)

    train_features, test_features, train_labels, test_labels = mlh.get_data(
        database=database,
        training_tables=training,
        validation_tables=validation,
        absolute=absolute,
        skip=skip,
        yval=yval,
    )

    train_features, test_features, train_labels, test_labels = train_features.values, test_features.values, train_labels.values, test_labels.values

    assert Nlayers == len(N_nodes)

    x = tf.placeholder(tf.float32, shape = [None, len(train_features[0])])
    y_ = tf.placeholder(tf.float32, shape = [None, 1])

    layer_dict = {}

    xconv = convolve(x, nfilters, convolution_outsize)

    for N in range(Nlayers):
        if N == 0:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(weights(x, convolution_outsize, N_nodes[N]))
        elif N + 1 == Nlayers:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(weights(layer_dict["layer_{}".format(N-1)], N_nodes[N-1], N_nodes[N]))
        else:
            layer_dict["layer_{}".format(N)] = tf.nn.elu(weights(layer_dict["layer_{}".format(N-1)], N_nodes[N-1], N_nodes[N]))

    y_out = layer_dict["layer_{}".format(Nlayers-1)]

    cost = tf.losses.huber_loss(labels = y_, predictions = y_out)
    #cost = tf.losses.mean_squared_error(labels = y_, predictions = y_out)

    #training_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    optimizer = tf.train.AdamOptimizer(0.01)
    training_step = optimizer.minimize(cost)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    training_error = []

    for _ in range(int(training_iterations)):
        batch_vectors, batch_answers = get_batch(train_features, train_labels)
        session.run(training_step,feed_dict={x:batch_vectors,y_:batch_answers})
        if _ % (int(training_iterations)//10) == 0:

            train_error = session.run(cost,feed_dict={x:train_features,y_:train_labels})
            training_error.append([_, train_error])
            print(train_error)

    pred_y = session.run(y_out, feed_dict={x: test_features})

    #rmse = session.run(cost,feed_dict={x:test_features,y_:test_labels})
    mae = np.mean(abs(pred_y-test_labels))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        test_labels.flatten(), pred_y.flatten()
    )

    #error_dictionary = find_largest_deviations(chromo_IDs, pred_y, test_labels)

    #plot_error_hist(error_dictionary)
    plot_comparison(test_labels, abs(pred_y), mae, r_value**2)

    saver.save(session, "./p3ht_brain")

def brain(database="p3ht.db", 
        absolute=None, 
        skip=[], 
        yval="TI", 
        training=None, 
        validation=None,
        Nlayers = 3,
        node_comb = [10, 5, 1],
        steps = 2e4,
        run_name = "",
        forward_hops_only = False,
        show_comparison = False, 
        ):

    run_net(database = database,
            training = training,
            validation = validation,
            absolute = absolute,
            skip = skip,
            yval = yval,
            Nlayers = Nlayers, 
            N_nodes= node_comb, 
            training_iterations = steps, 
            run_name = run_name, 
            show_comparison = show_comparison, 
            forward_hops_only = forward_hops_only)
