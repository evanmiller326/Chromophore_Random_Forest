import tensorflow as tf
import numpy as np
import sqlite3

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

def get_data(database = 'p3ht.db', ratio = 0.95):

    data = load_database(database)
    data = data[:,2:]

    data = shuffle_data(data)

    training = data[:int(len(data)*ratio),:]
    validation = data[int(len(data)*(ratio)):,:]

    return training[:,:-1], np.array([training[:,-1]]).T, validation[:,:-1], np.array([validation[:,-1]]).T

def weights(input_vector, in_nodes, out_nodes):
    W = tf.random_normal(shape=[in_nodes, out_nodes], stddev=0.3)
    b = tf.random_normal(shape=[out_nodes], stddev = 0.3)
    return build_layer(input_vector, tf.Variable(W), tf.Variable(b))

def get_batch(vectors, labels):
    indices = np.arange(len(vectors))
    np.random.shuffle(indices)
    chosen = indices[:5000]
    return vectors[chosen], labels[chosen]

def build_layer(x, W, b):
    return tf.add(tf.matmul(x,W),b)

def plot_comparison(actual, predicted):
    plt.scatter(actual, predicted, zorder=0, alpha = 0.5, s=12)
    plt.plot(np.linspace(0, np.amax(actual), 10), np.linspace(0, np.amax(actual), 10), c='k', zorder = 10)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig("Ann_comp.png")

def ANN(Nlayers = 1, N_nodes= [1], training_iterations = 5e4, run_name = "", show_comparison = False):

    training_vectors, training_answers, validation_vectors, validation_answers = get_data(database = 'p3ht.db', ratio = 0.95)

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

    optimizer = tf.train.AdamOptimizer(0.001)
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

    plot_comparison(validation_answers, pred_y)
    
if __name__ == "__main__":
    Nlayers = 4
    node_comb = [7, 20, 10, 1]
    steps = 5e4
    ANN(Nlayers = Nlayers, N_nodes= node_comb, training_iterations = steps)
