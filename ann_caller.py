import argparse
import ANN as ANN

def cli_tool():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="p3ht.db",
        required=False,
        help="""Select a database in this directory to use
                        for the random forest. Default ==
                        'p3ht.db'""",
    )
    parser.add_argument(
        "-a",
        "--absolute",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="""When column names are passed, random_forest.py
                        will exploit symmetries in the
                        chromophores, and only consider the
                        absolute values of the specified
                        descriptors. E.g. -a rotX rotY""",
    )
    parser.add_argument(
        "-s",
        "--skip",
        nargs="+",
        type=str,
        default=[],
        required=False,
        help="""When column names are passed, random_forest.py
                        will skip over these descriptors, meaning
                        that the ML will not train on them.
                        E.g. -s DeltaE""",
    )
    parser.add_argument(
        "-y",
        "--yval",
        type=str,
        default="TI",
        required=False,
        help="""When specified, use a feature other than "TI"
                        to train on and fit to.""",
    )
    parser.add_argument(
        "-t",
        "--training",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="""Set the table names from the database to be
                        used as the training data. E.g. -t
                        T1_5 T1_75 T2_0""",
    )
    parser.add_argument(
        "-v",
        "--validation",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="""Set the table names from the database to be
                        used as the validation data. E.g. -v
                        T1_5 T1_75 T2_0""",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli_tool()
    opts = vars(args)
    ANN.brain(**opts)
