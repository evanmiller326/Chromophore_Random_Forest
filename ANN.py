import tensorflow as tf
import numpy as np
import sqlite3

import sklearn
from sklearn.preprocessing import normalize
from sklearn.preprocessing import normalize
from sklearn import preprocessing 

import matplotlib.pyplot as plt

def load_database(database):
    data_list = []
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    cursor.execute(query)
    data = cursor.fetchall()
    for tab in data:
        query = "select * from {};".format(tab[0])
        cursor.execute(query)
        data = cursor.fetchall()
        data = np.array(data)
        for datum in data:
            data_list.append(datum)
    cursor.close()
    connection.close()
    data = np.array(data_list)
    return data

def shuffle_data(data):
    p = np.random.permutation(len(data))
    return data[p]

def get_data(database = 'p3ht.db', ratio = 0.95, forward_hops_only = False):

    data = load_database(database)
    data = shuffle_data(data)

    chromo_IDs = data[:,:2]
    data = data[:,2:]

    if forward_hops_only is True:
        print("ONLY CONSIDERING FORWARD HOPS")
        chromo_IDs = list(chromo_IDs)
        data = list(data)
        indices_to_delete = []
        for index, chromo_ID_pair in enumerate(chromo_IDs):
            if chromo_ID_pair[0] > chromo_ID_pair[1]:
                indices_to_delete.append(index)
        indices_to_delete = sorted(indices_to_delete, reverse=True)
        for index in indices_to_delete:
            chromo_IDs.pop(index)
            data.pop(index)
        chromo_IDs = np.array(chromo_IDs)
        data = np.array(data)

    #data = shuffle_data(data)

    features = data[:,:-1]
    answers = np.array([data[:,-1]]).T

    #scaler = preprocessing.StandardScaler().fit(features)
    #features = scaler.transform(features)
    features = normalize(features)

    cut_off = int(len(data)*ratio)

    return features[:cut_off,:], answers[:cut_off,:], features[cut_off:,:], answers[cut_off:,:], chromo_IDs

def weights(input_vector, in_nodes, out_nodes):
    W = tf.random_normal(shape=[in_nodes, out_nodes], mean=0.2, stddev=0.2)
    b = tf.random_normal(shape=[out_nodes], mean = 0.2, stddev = 0.2)
    #W = tf.random_normal(shape=[in_nodes, out_nodes], stddev=0.1)
    #b = tf.random_normal(shape=[out_nodes], stddev = 0.1)
    return build_layer(input_vector, tf.Variable(W), tf.Variable(b))

def get_batch(vectors, labels):
    indices = np.arange(len(vectors))
    np.random.shuffle(indices)
    chosen = indices[:5000]
    return vectors[chosen], labels[chosen]

def build_layer(x, W, b):
    return tf.add(tf.matmul(x,W),b)

def plot_comparison(actual, predicted, rmse):
    plt.scatter(actual, predicted, zorder=0, alpha = 0.5, s=12)
    plt.plot(np.linspace(0, np.amax(actual), 10), np.linspace(0, np.amax(actual), 10), c='k', zorder = 10)
    plt.title("ANN-RMSE-{:.5f}".format(rmse))
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig("Ann_comp.png")

def find_largest_deviations(chromo_IDs, pred_y, actual_y):
    differences = np.array(actual_y) - np.array(pred_y)
    dictionary = {}
    error_dictionary = {diff[0]: chromo_IDs[index] for index, diff in enumerate(list(differences))}
    sorted_keys = sorted(error_dictionary.keys())
    print("Largest underestimations =")
    for i in range(10):
        print("Chromos =", error_dictionary[sorted_keys[i]], "difference =", sorted_keys[i])
    print("\nLargest overestimations =")
    for i in range(1, 11):
        print("Chromos =", error_dictionary[sorted_keys[-i]], "difference =", sorted_keys[-i])
    return error_dictionary


def plot_error_hist(error_dictionary):
    plt.figure()
    plt.hist(list(error_dictionary.keys()), bins=100)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.xlim([-0.5, 0.5])
    plt.savefig("./error_histogram_all.png")

def ANN(Nlayers = 1, N_nodes= [1], training_iterations = 5e4, run_name = "", show_comparison = False, forward_hops_only = False):

    training_vectors, training_answers, validation_vectors, validation_answers, chromo_IDs = get_data(database = 'p3ht.db', ratio = 0.95, forward_hops_only = forward_hops_only)

    assert Nlayers == len(N_nodes)


    x = tf.placeholder(tf.float32, shape = [None, len(training_vectors[0])])
    y_ = tf.placeholder(tf.float32, shape = [None, 1])

    layer_dict = {}

    for N in range(Nlayers):
        if N == 0:
            layer_dict["layer_{}".format(N)] = tf.nn.relu(weights(x, len(training_vectors[0]), N_nodes[N]))
        elif N + 1 == Nlayers:
            layer_dict["layer_{}".format(N)] = tf.nn.relu(weights(layer_dict["layer_{}".format(N-1)], N_nodes[N-1], N_nodes[N]))
        else:
            layer_dict["layer_{}".format(N)] = tf.nn.relu(weights(layer_dict["layer_{}".format(N-1)], N_nodes[N-1], N_nodes[N]))

    y_out = layer_dict["layer_{}".format(Nlayers-1)]

    cost = tf.losses.mean_squared_error(labels = y_, predictions = y_out)

    #training_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    optimizer = tf.train.AdamOptimizer(0.01)
    training_step = optimizer.minimize(cost)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    training_error = []

    for _ in range(int(training_iterations)):
        batch_vectors, batch_answers = get_batch(training_vectors, training_answers)
        session.run(training_step,feed_dict={x:batch_vectors,y_:batch_answers})
        if _ % (int(training_iterations)//10) == 0:

            train_error = session.run(cost,feed_dict={x:training_vectors,y_:training_answers})
            training_error.append([_, train_error])
            print(train_error)

    #batch_vectors, batch_answers = get_batch(validation_vectors, validation_answers)

    pred_y = session.run(y_out, feed_dict={x: validation_vectors})

    rmse = session.run(cost,feed_dict={x:validation_vectors,y_:validation_answers})

    error_dictionary = find_largest_deviations(chromo_IDs, pred_y, validation_answers)

    #plot_error_hist(error_dictionary)
    plot_comparison(validation_answers, pred_y, rmse)


if __name__ == "__main__":
    Nlayers = 2
    node_comb = [9, 1]
    steps = 2e4
    forward_hops_only = False
    ANN(Nlayers = Nlayers, N_nodes= node_comb, training_iterations = steps, forward_hops_only = forward_hops_only)
