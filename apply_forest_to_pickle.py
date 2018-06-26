import numpy as np
import extract_data_from_pickles as ep
from sklearn.externals import joblib
from morphct.code import helper_functions as hf
from morphct.utils import KMC_analyse as kmc_a
import sys
import sqlite3
import os
import argparse
from glob import glob
from shutil import copyfile
import time

def predict_transfer_integrals(pickle, forest, species, molecule, overwrite, training_metrics, absolute):

    start_time = time.time()

    try:
        pickle_backup = pickle+".bak"
        copyfile(pickle, pickle_backup)
    except:
        print("Could not make backup file. Exiting!")

    #Load in the data
    AA_morphology_dict, CG_morphology_dict, CG_to_AAID_master, parameter_dict, chromophore_list = hf.load_pickle(pickle)

    molecule_dict = ep.get_molecule_dictionary(molecule)

    species_list = [chromophore.species for chromophore in chromophore_list]
    chromo_IDs = [index for index, chromo_species in enumerate(species_list) if chromo_species == species]
    vector_dict = ep.generate_empty_dict(chromo_IDs)
    box = np.array([[AA_morphology_dict['lx'], AA_morphology_dict['ly'], AA_morphology_dict['lz']]])

    print("Generating Chromophore Descriptors")
    vector_dict = ep.fill_dict(vector_dict, 
            chromophore_list, 
            AA_morphology_dict, 
            box, 
            species, 
            molecule_dict['atom_indices'])
    mol_lookup_dict = ep.identify_chains(AA_morphology_dict, chromophore_list)

    print("Iterating Through Chromophores")
    for i, chromophore in enumerate(chromophore_list):
        #Only get the desired acceptor or donor index, needed for blends.
        if chromophore.species == species:
            #Iterate through the neighbors of each chromophore.
            for sub_index, (neighbor, n_deltaE, n_TI) in enumerate(zip(chromophore.neighbours, chromophore.neighbours_delta_E, chromophore.neighbours_TI)):
                inputs = {}
                index1 = i
                index2 = neighbor[0]  # Gets the neighbor's index
                relative_image = neighbor[1] # Gets the neighbour's image
                inputs['deltaE'] = n_deltaE  # Get the difference in energy
                TI = n_TI  # Get the transfer integral

                inputs['same_chain'] = mol_lookup_dict[index1] == mol_lookup_dict[index2]
                if os.path.splitext(molecule_dict['database'])[0].lower() == 'p3ht':
                    inputs['sulfur_distance'] = ep.get_sulfur_separation(chromophore, chromophore_list[index2], relative_image, box[0], AA_morphology_dict)
                else:
                    inputs['sulfur_distance'] = 0

                if (TI is None) or overwrite == True:  # Consider only pairs that will have hops.
                    #Get the location of the other chromophore and make sure they're in the
                    #same periodic image.
                    index2_loc = ep.check_vector(chromophore_list[index2].posn, chromophore.posn, box)
                    #Calculate the distance (normally this is in Angstroms)
                    centers_vec = chromophore.posn - index2_loc

                    #Get the vectors describing the two chromophores
                    vdict1 = vector_dict[index1]
                    vdict2 = vector_dict[index2]
                    #Calculate the differences in alignment between the chromophores
                    # Vec1 for DBP is (midpoint(1, 2) -> 3) (along Y axis)
                    # Vec2 for DBP is (1 -> 2) (along X axis)
                    # Vec3 for DBP is Normal to both (along Z axis)
                    # MJ note: FOR DBP ONLY if we assume the chromophore is identical above
                    # and below the plane of the molecule, then because
                    # dot(a, b) == - dot(a, -b), we can reduce the dimensionality
                    # of our data further by just taking the absolute value of the
                    # dot product. THIS IS NOW INCLUDED IN RANDOM_FOREST.PY as
                    # a toggleable flag.
                    inputs['rotY'] = np.dot(vdict1['vec1'], vdict2['vec1'])
                    inputs['rotX'] = np.dot(vdict1['vec2'], vdict2['vec2'])
                    inputs['rotZ'] = np.dot(vdict1['vec3'], vdict2['vec3'])

                    # We need to calculate a rotation matrix that can map the normal vector
                    # describing the plane of the first chromophore to the z-axis.
                    # We can define this rotation matrix as the required transformation to
                    # map the normal of chromophore 1 (a) to the unit z axis (b).
                    rotation_matrix = ep.calculate_rotation_matrix(vdict1['vec3'], vdict1['vec2'])
                    old_length = np.linalg.norm(centers_vec)
                    transformed_separation_vec = np.matmul(rotation_matrix, centers_vec)
                    new_length = np.linalg.norm(transformed_separation_vec)
                    assert np.isclose(old_length, new_length)

                    inputs['posX'] = centers_vec[0]
                    inputs['posY'] = centers_vec[1]
                    inputs['posZ'] = centers_vec[2]
                    #Combine the data into an array

                    for metric in absolute:
                        try:
                            inputs[metric] = abs(inputs[metric])
                        except:
                            print("Tried to take absolute of a metric not in the list.")

                    train_array = np.array([[inputs[metric] for metric in list(sorted(training_metrics))]])
                    predicted_value = forest.predict(train_array)[0]

                    chromophore.neighbours_TI[sub_index] = predicted_value

    print("Run Time:", (time.time()-start_time)/60)

    hf.write_pickle((AA_morphology_dict, CG_morphology_dict, CG_to_AAID_master, parameter_dict, chromophore_list), pickle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--species", required=False,
            help="""Pass this flag so that only pairs
            of the specified species (donor or acceptor)
            are added into the database.
            If flag is not passed, defaults to 'donor'.
            """)
    parser.add_argument("-p", "--pickles", required=True,
            #I know requiring an argument is bad, may want to change later.
            default=None,
            type=str,
            nargs='+',
            help="""Pass this flag to specify a 
            pickles to load and input data into.
            """)
    parser.add_argument("-o", "--overwrite", required=False,
            action='store_true',
            help="""Pass this flag to specify 
            if a pickle with TIs should be overwritten.
            """)
    parser.add_argument("-f", "--forest", required=False,
            help="""Pass this flag to specify 
            the random forest pickle to load and use
            to predict the transfer integrals.
            """)
    parser.add_argument("-m", "--molecule", required=True,
            help="""Pass this flag to specify the
            molecule directory to look in for the
            training data.
            """)
    parser.add_argument(
        "-a",
        "--absolute",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="""When column names are passed, the absolute
                        volumns of the columns will be used.
                        E.g. -a rotX rotY""",
    )
    parser.add_argument(
        "-ex",
        "--exclude",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="""Set the metrics that will be excluded
        to prdict the transfer integrals.""",
    )

    args, input_list = parser.parse_known_args()

    training_metrics = ['posX', 'posY', 'posZ', 'rotX', 'rotY', 'rotZ', 'deltaE', 'same_chain', 'sulfur_distance']
    if args.exclude != None:
        for metric in exlude:
            try:
                training_metrics.pop(metric)
            except:
                print("Tried to exclude metric not found in training_metrics!")

    absolute = ['posX', 'posY', 'posZ', 'rotX', 'rotY', 'rotZ', 'deltaE']
    if args.absolute:
        absolute = args.absolute

    overwrite = False
    if args.overwrite:
        overwrite = args.overwrite

    forest = joblib.load("saved_random_forest.pkl")
    if args.forest:
        forest = joblib.load(args.forest)

    molecule = args.molecule.lower()

    species = 'donor'
    if args.species:
        species = args.species

    for pickle in args.pickles:
        predict_transfer_integrals(pickle, forest, species, molecule, overwrite, training_metrics, absolute)

if __name__ == "__main__":
    main()
