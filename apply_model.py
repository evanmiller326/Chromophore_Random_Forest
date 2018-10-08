import matplotlib
matplotlib.use('agg')
from sklearn.externals import joblib
import ml_helpers as mlh
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
from pandas.plotting import parallel_coordinates

def plot_parallel(df, table):
    plt.close()
    df = df.drop(["chromophoreA","chromophoreB"], axis=1)
    parallel_coordinates(df, 'errors', colormap = 'viridis')
    plt.title(table)
    plt.savefig("{}-parallel.png")

def plot_actual_vs_predicted(test_labels, predictions, r_value, abserr, test_features,name=""):
    for column in test_features.columns.values.tolist():
        colors = test_features[[column]]
        plt.close()
        plt.plot(
            np.linspace(0, np.amax(test_labels.values), 10),
            np.linspace(0, np.amax(test_labels.values), 10),
            alpha=0.5,
            c="k",
            zorder=10,
            label=r"R$^2$={:.3f}, MAE={:.0f} meV".format(r_value ** 2, abserr * 1000),
        )
        plt.scatter(test_labels.values, predictions, s=12, alpha=0.5, c=colors.values, zorder=0, cmap='viridis')
        plt.colorbar()
        plt.legend(fontsize=20)
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])

        plt.xlabel("Actual TI (eV)")
        plt.ylabel("Predicted TI (eV)")

        plt.title("{}-{}".format(name,column), fontsize = 12)

        plt.savefig("images/{}-{}-comparison.png".format(name, column))

def run_table(table, reg):

    temp_features, test_features, train_labels, test_labels = mlh.get_data(database="p3ht_pdi.db", training_tables=[table], validation_tables=[table], absolute=["rotX", "rotY", "rotZ", "posX", "posY", "posZ", "deltaE"], skip=["TI"])

    test_features = test_features.drop(["chromophoreA","chromophoreB"], axis=1)
    print("making temp")

    temp_features["bonded"] = abs(temp_features['chromophoreA']-temp_features['chromophoreB'])

    temp_features["bonded"] = [val if temp_features['bonded'][i] == 1 else 0 for i, val in enumerate(temp_features['same_chain'])]

    test_features['same_chain'] = temp_features['bonded']

    df = pd.concat([temp_features, train_labels], axis=1)

    feature_importances = pd.DataFrame(
        reg.feature_importances_, index=test_features.columns, columns=["Importance"]
    ).sort_values("Importance", ascending=False)

    predictions = np.array([reg.predict(test_features)]).T

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        test_labels.values.flatten(), predictions.flatten()
    )

    df["errors"] = predictions - test_labels

    df["predicted"] = df[["errors", "TI"]].sum(axis=1)

    rmse = mean_squared_error(test_labels, predictions)

    abserr = np.mean(abs(test_labels.values - predictions))

    plot_actual_vs_predicted(test_labels, predictions, r_value, abserr, temp_features, name=table+"-")

    #plot_parallel(df, table)

if __name__ == "__main__":
    reg = joblib.load('saved_random_forest.pkl') 
    table = sys.argv[1]
    run_table(table, reg)
