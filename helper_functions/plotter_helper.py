import matplotlib.pyplot as plt
from os.path import isdir
from os import mkdir
import numpy as np

def plot_actual_vs_predicted(
    actual, predicted, r_value=0.0, abserr=0.0, name="Act. v Pred.", save=True
):
    """
    actual: array for the actual transfer integrals
    predicted: array for the predicted TIs
    r_value: float for the correlelation coefficient
    mae: float for the mean absolute error
    name: string for plot title
    save: bool to save or show the plot
    """

    fig, ax = plt.subplots()
    ax.plot(
        np.linspace(0, np.amax(actual), 10),
        np.linspace(0, np.amax(actual), 10),
        alpha=0.5,
        c="k",
        zorder=10,
        label=r"R$^2$={:.2f}, MAE={:.0f} meV".format(r_value ** 2, abserr * 1000),
    )
    ax.scatter(actual, predicted, s=12, alpha=0.5, zorder=0)
    ax.legend(fontsize=20)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])

    ax.set_xlabel("Actual (eV)")
    ax.set_ylabel("Predicted (eV)")

    ax.set_title("{}".format(name))

    if save:
        if not isdir("images"):
            mkdir("./images")
        plt.savefig("images/{}comparison.pdf".format(name),
                bbox_inches='tight',
                transparent = True
                )
    else:
        plt.show()


def plot_based_on_density(actual, predicted, title, r_value=0.0, mae=0.0, save=True):
    """
    actual: array for the actual transfer integrals
    predicted: array for the predicted TIs
    title: string for plot title
    r_value: float for the correlelation coefficient
    mae: float for the mean absolute error
    save: bool to save or show the plot
    """
    from matplotlib.colors import LogNorm
    from matplotlib import cm

    print("Plotting Actual vs. Predicted 2D Map.")

    fig, ax = plt.subplots()

    rgba = cm.get_cmap("viridis")
    rgba = rgba(0.0)

    ax.set_facecolor(rgba)

    plt.hist2d(actual, predicted, (50, 50), norm=LogNorm(), cmap="viridis")
    plt.colorbar()

    ax.plot(
        np.linspace(0, np.amax(actual), 10),
        np.linspace(0, np.amax(actual), 10),
        "w--",
        alpha=0.5,
    )

    ax.text(
        0.8,
        0.1,
        s=r"R$^{2}$:" + "{:.4f}\nMAE:{:.4f}".format(r_value ** 2, mae),
        color="w",
        fontsize=20,
    )

    ax.set_xlabel("Actual (eV)")
    ax.set_ylabel("Predicted (eV)")
    ax.set_title("{}".format(title))

    ax.set_xlim([0, np.amax(actual)])
    ax.set_ylim([0, np.amax(predicted)])

    if save:
        if not isdir("images"):
            print("Note: Making an \"images\" directory to save figure.")
            mkdir("./images")
        plt.savefig("images/{}-comparison_map.pdf".format(title),
                bbox_inches='tight',
        #        transparent = True
                )
    else:
        plt.show()
