import argparse
import numpy as np
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.tree import export_graphviz
import pydot

from sklearn.externals import joblib


def get_data(database="p3ht.db", training_tables=None, validation_tables=None,
             absolute=None, skip=[], yval="TI"):
    training_records = []
    print("".join(["Loading data from ", database, "..."]))
    # Obtain training tables first:
    if training_tables is None:
        # Fetch all tables
        print("Using all tables to train from...")
        connection = sqlite3.connect(database)
        cursor = connection.cursor()
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        cursor.execute(query)
        training_tables = [name[0] for name in cursor.fetchall()]
        cursor.close()
        connection.close()
    for table_name in training_tables:
        data, all_column_names = load_table(database, table_name)
        for record in data:
            training_records.append(record)
    column_names_to_use = list(set(all_column_names) - set(skip))
    train_features, train_labels = create_data_frames(np.array(training_records),
                                                      absolute,
                                                      all_column_names,
                                                      column_names_to_use,
                                                      yval)
    print("Separating training and test data...")
    if validation_tables is None:
        # Split the dataset we have to be 95%:5%
        return train_test_split(train_features, train_labels, test_size=0.05)
    else:
        # Need to load the right validation data:
        validation_records = []
        for table_name in validation_tables:
            data, _ = load_table(database, table_name)
            for record in data:
                validation_records.append(record)
        test_features, test_labels = create_data_frames(np.array(validation_records),
                                                        absolute,
                                                        all_column_names,
                                                        column_names_to_use,
                                                        yval)

    test_features.sort_index(axis=1, inplace=True)
    train_features.sort_index(axis=1, inplace=True)

    return train_features, test_features, train_labels, test_labels


def create_data_frames(data, absolute, all_column_names, column_names_to_use,
                       yval):
    df = pd.DataFrame(data, columns=all_column_names)

    if absolute is not None:
        for col_name in absolute:
            df[col_name] = df[col_name].abs()
    features = df[column_names_to_use]
    y = df[[yval]]
    return features, y


def load_table(database, table):
    data_list = []
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    query = "SELECT * FROM {};".format(table)
    cursor.execute(query)
    data = cursor.fetchall()
    data = np.array(data)
    cursor.execute("PRAGMA table_info({})".format(table))
    column_names = [col_props[1] for col_props in cursor.fetchall()]
    cursor.close()
    connection.close()
    return np.array(data), column_names


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
    # reg = RandomForestRegressor(max_features = 'sqrt', min_samples_leaf = 1, bootstrap=True, max_depth=30, min_samples_split=5, n_estimators = 733)
    print("Starting Training.")
    reg.fit(train_features, train_labels)
    return reg


def get_biggest_discrepencies(predicted, actual, errors):
    comparison = []
    for pair in zip(predicted, actual):
        comparison.append([pair[1], pair[0], abs(pair[1] - pair[0])])
    comparison = list(sorted(comparison, key=lambda x: x[2]))
    print(comparison)


def make_plots(test_labels, predictions, test_features, features):
    for feature in features:
        plt.close()
        colors = abs(test_features[feature].values)
        colors /= np.amax(colors)
        plt.scatter(
            test_labels, predictions, s=12, alpha=0.5, zorder=0, c=colors, cmap="jet"
        )
        plt.plot(
            np.linspace(0, np.amax(test_labels.values), 10),
            np.linspace(0, np.amax(test_labels.values), 10),
            c="k",
            zorder=10,
        )
        plt.colorbar()
        plt.ylabel("Predicted TI")
        plt.xlabel("Actual TI")
        plt.title("{}".format(feature))
        plt.savefig("{}_comparison.png".format(feature))

def plot_actual_vs_predicted(test_labels, predictions, r_value, abserr, name=""):
    plt.close()
    plt.plot(
        np.linspace(0, np.amax(test_labels.values), 10),
        np.linspace(0, np.amax(test_labels.values), 10),
        alpha=0.5,
        c="k",
        zorder=10,
        label=r"R$^2$={:.3f}, MAE={:.0f} meV".format(r_value ** 2, abserr * 1000),
    )
    plt.scatter(test_labels.values, predictions, s=12, alpha=0.5, zorder=0)
    plt.legend(fontsize=20)
    plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])

    plt.xlabel("Actual TI (eV)")
    plt.ylabel("Predicted TI (eV)")

    plt.title("{}Act. v. Pred.".format(name))

    plt.savefig("{}comparison.png".format(name))
    plt.savefig("{}comparison.pdf".format(name))

def compare_four():
    # tables = ['cryst_frame_7', 'cryst_frame_11']
    tables = ["disordered_frame_10", "cryst_frame_7", "semicry_frame_10"]
    counter = 1
    plt.figure(figsize=(len(tables) * 3, len(tables) * 2.7))
    for table1 in tables:
        for table2 in tables:
            t1 = load_table(table1)
            df1 = pd.DataFrame(
                t1,
                columns=[
                    "Chromophore1",
                    "Chromophore2",
                    "posX",
                    "posY",
                    "posZ",
                    "rotX",
                    "rotY",
                    "rotZ",
                    "DeltaE",
                    "same_chain",
                    "sulfur_dist",
                    "TI",
                ],
            )

            df1["posX"] = df1["posX"].abs()
            df1["posY"] = df1["posY"].abs()
            df1["posZ"] = df1["posZ"].abs()
            df1["rotX"] = df1["rotX"].abs()
            df1["rotY"] = df1["rotY"].abs()
            df1["rotZ"] = df1["rotZ"].abs()

            if table1 == table2:

                features = df1[
                    [
                        "posX",
                        "posY",
                        "posZ",
                        "rotX",
                        "rotY",
                        "rotZ",
                        "DeltaE",
                        "same_chain",
                        "sulfur_dist",
                    ]
                ]
                y = df1[["TI"]]

                train_features, test_features, train_labels, test_labels = train_test_split(
                    features, y, test_size=0.05
                )

            else:

                t2 = load_table(table2)
                df2 = pd.DataFrame(
                    t2,
                    columns=[
                        "Chromophore1",
                        "Chromophore2",
                        "posX",
                        "posY",
                        "posZ",
                        "rotX",
                        "rotY",
                        "rotZ",
                        "DeltaE",
                        "same_chain",
                        "sulfur_dist",
                        "TI",
                    ],
                )

                df2["posX"] = df2["posX"].abs()
                df2["posY"] = df2["posY"].abs()
                df2["posZ"] = df2["posZ"].abs()
                df2["rotX"] = df2["rotX"].abs()
                df2["rotY"] = df2["rotY"].abs()
                df2["rotZ"] = df2["rotZ"].abs()

                train_features = df1[
                    [
                        "posX",
                        "posY",
                        "posZ",
                        "rotX",
                        "rotY",
                        "rotZ",
                        "DeltaE",
                        "same_chain",
                        "sulfur_dist",
                    ]
                ]
                train_labels = df1[["TI"]]

                test_features = df2[
                    [
                        "posX",
                        "posY",
                        "posZ",
                        "rotX",
                        "rotY",
                        "rotZ",
                        "DeltaE",
                        "same_chain",
                        "sulfur_dist",
                    ]
                ]
                test_labels = df2[["TI"]]

            reg = normal_run(train_features, train_labels)

            predictions = np.array([reg.predict(test_features)]).T
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                test_labels.values.flatten(), predictions.flatten()
            )

            rmse = mean_squared_error(test_labels, predictions)
            abserr = np.mean(abs(test_labels.values - predictions))

            plt.subplot(len(tables), len(tables), counter)
            plt.plot(
                np.linspace(0, np.amax(test_labels.values), 10),
                np.linspace(0, np.amax(test_labels.values), 10),
                linewidth=1,
                alpha=0.5,
                c="k",
                zorder=10,
                label="R^2={:.3f},\nMAE={:.0f} meV".format(r_value ** 2, abserr * 1000),
            )
            plt.scatter(test_labels, predictions, s=5, alpha=0.5, zorder=0)
            plt.title("{}-{}".format(table1, table2), fontsize=7)
            plt.legend(fontsize=6, handlelength=0)
            plt.ylabel("Predicted TI (eV)", fontsize=7)
            plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], fontsize=7)
            plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], fontsize=7)
            plt.xlabel("Actual TI (eV)", fontsize=7)
            counter += 1
            plt.savefig("{}_comparison.png".format(len(tables) ** 2))
            del reg, train_features, test_features, train_labels, test_labels

def model_analysis(predictions, test_labels, df):
    '''
    Measure how well the model performs
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        test_labels.values.flatten(), predictions.flatten()
    )
    print("Calculating error.")
    df["errors"] = predictions - test_labels

    df["predicted"] = df[["errors", "TI"]].sum(axis=1)

    print("Mean Deviation =", df["errors"].mean())
    print(df.nlargest(10, "errors"))

    #rmse = mean_squared_error(test_labels, predictions)
    abserr = np.mean(abs(test_labels.values - predictions))

    return abserr, r_value

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

    # compare_four()

    train_features, test_features, train_labels, test_labels = get_data(
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
    df = pd.concat([df_train, df_test], axis=0)

    reg = normal_run(train_features, train_labels)

    joblib.dump(reg, "saved_random_forest.pkl")

    print(reg.get_params())

    print("Making predictions on tests")
    predictions = np.array([reg.predict(test_features)]).T

    abserr, r_value = model_analysis(predictions, test_labels, df)

    plot_actual_vs_predicted(test_labels, predictions, r_value, abserr)

    # make_plots(test_labels, predictions, test_features, features)

    feature_importances = pd.DataFrame(
        reg.feature_importances_, index=train_features.columns, columns=["Importance"]
    ).sort_values("Importance", ascending=False)

    # get_biggest_discrepencies(predictions, test_labels, errors)

    # df.hist(bins = 50, figsize = (14, 12))
    # plt.savefig('hist.pdf')
    # plt.close()
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
