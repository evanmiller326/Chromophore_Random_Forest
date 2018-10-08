import numpy as np
import pandas as pd
import sqlite3

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
    df = df.sort_index(axis=1)

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

