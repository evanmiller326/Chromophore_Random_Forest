import numpy as np
import sqlite3 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz
import pydot

def get_data():
    data_list = []
    connection = sqlite3.connect("dbp.db")
    #connection = sqlite3.connect("p3ht.db")
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

def plot_tree(reg):
    print("Building Tree.")
    tree = reg.estimators_[5]

    feature_list = list(train_features.columns)
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')

def random_search(train_features, test_features, train_labels, test_labels):
    from sklearn.model_selection import RandomizedSearchCV

    reg = RandomForestRegressor()

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    reg_random = RandomizedSearchCV(estimator = reg, 
            param_distributions = random_grid, 
            n_iter = 50, 
            cv = 3, 
            verbose=2, 
            random_state=42, 
            n_jobs = -1)

    reg_random.fit(train_features, train_labels)
    print(reg_random.best_params_)
    return reg

def normal_run(train_features, train_labels):
    print("Setting up the regression.")
    reg = RandomForestRegressor()
    #reg = RandomForestRegressor(max_features = 'sqrt', min_samples_leaf = 1, bootstrap=True, max_depth=30, min_samples_split=5, n_estimators = 733)
    print("Starting Training.")
    reg.fit(train_features, train_labels)
    return reg

def get_biggest_discrepencies(predicted, actual, errors):
    comparison = []
    for pair in zip(predicted, actual):
        comparison.append([pair[1], pair[0], abs(pair[1]-pair[0])])
    comparison = list(sorted(comparison,key=lambda x: x[2]))
    print(comparison)

def make_plots(test_labels, predictions, test_features, features):
    for feature in features:
        plt.close()
        colors = abs(test_features[feature].values)
        colors /= np.amax(colors)
        plt.scatter(test_labels, predictions, s= 12, alpha = 0.5, zorder=0, c = colors, cmap='jet')
        plt.plot(np.linspace(0, np.amax(test_labels.values), 10), np.linspace(0, np.amax(test_labels.values), 10), c='k', zorder=10)
        plt.colorbar()
        plt.ylabel("Predicted TI")
        plt.xlabel("Actual TI")
        plt.title("{}".format(feature))
        plt.savefig("{}_comparison.png".format(feature))


if __name__ == "__main__":
    print("Initializing Data.")
    data = get_data()
    print(data.shape)
    df = pd.DataFrame(data, columns = ['Chromophore1', 
        'Chromophore2', 
        'distance',
        'posX',
        'posY',
        'posZ',
        'rotX',
        'rotY',
        'rotZ',
        'DeltaE', 
        'TI'])

    #print("Making dot products absolutes values.")
    #df['dotB'] = df['dotB'].abs()
    #df['dotC'] = df['dotC'].abs()
    df = df.drop(['Chromophore1', 'Chromophore2'], axis=1)

    print("Splitting inputs and outputs")
    features = df[['posX', 'posY', 'posZ', 'rotX', 'rotY', 'rotZ', 'DeltaE']]
    y = df[['TI']]
    print(features.shape, y.shape)

    print("Separating training and test data.")
    train_features, test_features, train_labels, test_labels = train_test_split(features, y, test_size = 0.05)

    reg = normal_run(train_features, train_labels)

    print(reg.get_params())

    print("Making predictions on tests")
    predictions = np.array([reg.predict(test_features)]).T

    print("Calculating error.")
    df['errors'] = predictions - test_labels
    df['predicted'] = df[['errors', 'TI']].sum(axis=1)

    print(df.nlargest(10, 'errors'))

    #print(df.loc[df['Chromophore1'].isin([10204, 10205, 10206, 10207]) & df['Chromophore2'].isin([10046, 10047, 10048, 10049])])

    plt.close()
    plt.scatter(test_labels, predictions, s= 12, alpha = 0.5, zorder=0)
    plt.plot(np.linspace(0, np.amax(test_labels.values), 10), np.linspace(0, np.amax(test_labels.values), 10), c='k', zorder=10)
    plt.ylabel("Predicted TI")
    plt.xlabel("Actual TI")
    plt.savefig("comparison.png")

    make_plots(test_labels, predictions, test_features, features)

    feature_importances = pd.DataFrame(reg.feature_importances_,
            index = features.columns,
            columns = ['Importance']).sort_values('Importance', ascending=False)

    #get_biggest_discrepencies(predictions, test_labels, errors)

    #df.hist(bins = 50, figsize = (14, 12))
    #plt.savefig('hist.pdf')
    #plt.close()
    corr_matrix = df.corr()
    print(corr_matrix['TI'].sort_values(ascending=False))
    #scatter_matrix(df, figsize = (12,12), alpha = 0.2)
    #plt.savefig('scatter_matrix.png')
