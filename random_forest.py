import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import pydot

from sklearn.externals import joblib

from helper_functions import load_data_helper as ldh
from helper_functions import general_helpers as gh
from helper_functions import plotter_helper as ph


def plot_tree(reg):
    print("Building Tree.")
    tree = reg.estimators_[5]

    feature_list = list(train_features.columns)
    export_graphviz(
        tree, out_file="tree.dot", feature_names=feature_list, rounded=True, precision=1
    )
    # Use dot file to create a graph
    (graph,) = pydot.graph_from_dot_file("tree.dot")
    # Write graph to a png file
    graph.write_png("tree.png")


def random_search(train_features, test_features, train_labels, test_labels):
    from sklearn.model_selection import RandomizedSearchCV

    reg = RandomForestRegressor()

    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
    max_features = ["auto", "sqrt"]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    reg_random = RandomizedSearchCV(
        estimator=reg,
        param_distributions=random_grid,
        n_iter=50,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    reg_random.fit(train_features, train_labels)
    print(reg_random.best_params_)
    return reg


def normal_run(train_features, train_labels):
    print("Setting up the regression.")
    reg = RandomForestRegressor()
#    reg = RandomForestRegressor(max_features = 'sqrt', min_samples_leaf = 1, bootstrap=True, max_depth=30, min_samples_split=5, n_estimators = 733)
    print("Starting Training.")
    reg.fit(train_features, train_labels)
    return reg


def get_biggest_discrepencies(predicted, actual, errors):
    comparison = []
    for pair in zip(predicted, actual):
        comparison.append([pair[1], pair[0], abs(pair[1] - pair[0])])
    comparison = list(sorted(comparison, key=lambda x: x[2]))
    print(comparison)


def model_analysis(predictions, test_labels, df):
    '''
    Measure how well the model performs
    '''

    r_value, mae = gh.calc_comparison(test_labels.values, predictions)

    print("Calculating error.")
    df["errors"] = predictions - test_labels

    df["predicted"] = df[["errors", "TI"]].sum(axis=1)

    df["errors"] = df['errors'].abs()

    print("Mean Deviation =", df["errors"].mean())
    print(df.nlargest(40, "errors"))

    return mae, r_value

# Needs a fun name
def wood_chipper(database="p3ht.db", absolute=None, skip=[], yval="TI", training=None, validation=None):

    # Don't want to train on the chromophore_IDs!
    chromophore_ID_cols = ["chromophoreA", "chromophoreB"]
    for chromophore_ID_col in chromophore_ID_cols:
        if chromophore_ID_col not in skip:
            skip.append(chromophore_ID_col)
    # Also don't want to train on the answer!
    if yval not in skip:
        skip.append(yval)

    train_features, test_features, train_labels, test_labels = ldh.get_data(
        database=database,
        training_tables=training,
        validation_tables=validation,
        absolute=absolute,
        skip=skip,
        yval=yval,
    )

    # Concatenate the entire dataset
    # Add the TI columns by concatenating along axis=1
    df_train = pd.concat([train_features, train_labels], axis=1)
    df_test = pd.concat([test_features, test_labels], axis=1)
    # Stitch the training and testing data by concatenating along
    # axis = 0
    df = df_test
    #df = pd.concat([df_train, df_test], axis=0)

    print(train_features)

    reg = normal_run(train_features, train_labels)

    joblib.dump(reg, "saved_random_forest.pkl")

    print(reg.get_params())

    print("Making predictions on tests")
    predictions = np.array([reg.predict(test_features)]).T

    mae, r_value = model_analysis(predictions, test_labels, df)

    ph.plot_based_on_density(test_labels.flatten(), predictions.flatten(), "ANN", r_value**2, mae, save=True)

    feature_importances = pd.DataFrame(
        reg.feature_importances_, index=train_features.columns, columns=["Importance"]
    ).sort_values("Importance", ascending=False)

    # get_biggest_discrepencies(predictions, test_labels, errors)

    corr_matrix = df.corr()
    print(corr_matrix["TI"].sort_values(ascending=False))
    # scatter_matrix(df, figsize = (12,12), alpha = 0.2)
    # plt.savefig('scatter_matrix.png')

    # This will overwrite your old store.h5 file!
    with pd.HDFStore('store.h5', mode='w') as store:
        store['train_features'] = train_features
        store['test_features'] = test_features
        store['train_labels'] = train_labels
        store['test_labels'] = test_labels
        store['df'] = df
