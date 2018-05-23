import numpy as np
from morphct.code import helper_functions as hf
import sys
import sqlite3
import os
import argparse
from glob import glob

def n_unwrapped(positions, box):
    """
    Counts if the periodic
    boundaries needs to be applied to fully
    unwrap the positions.
    Arguments:
        positions - numpy array
        box - tuple or list
    Returns:
        Integer
    """
    s = 0
    for i in range(3):
        s += np.sum(np.absolute(positions[i]) > box[i]/2.)
    return s

def pbc(positions, box):
    box = box[0]
    """
    Wraps positions into a periodic box.
    Ensures neighbors across a periodic boundary are calculated
    as being next to eachother.
    Arguments:
        positions - numpy array (assumes just three coordinates of one particle)
        box - tuple or list, should have three elements
              assumes box goes from -L/2 to L/2.
    Returns: array of wrapped coordiantes
    """
    p = np.copy(positions)
    for i in range(3):
        mask = np.absolute(p[i]) > box[i]/2.
        if mask:
            p[i] -= np.sign(p[i])*box[i]
    if n_unwrapped(p, box) == 0:
        return p
    else:
        return pbc(p, box)

def create_data_base(database, systems):
    """
    Create an empty sql database.
    Arguments:
        database - string for the database name
    """
    try:
        os.remove(database)
    except:
        print("No {}. Creating New.".format(database))
        pass

    tables = [key for key, pair in systems.items()]
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    for table in tables:
        to_execute = "CREATE TABLE {}( ".format(table)
        to_execute += "chromophoreA INT, "
        to_execute += "chromophoreB INT, "
        to_execute += "distance REAL, "
        to_execute += "deltaE REAL, "
        to_execute += "alignX REAL, "
        to_execute += "alignY REAL, "
        to_execute += "alignZ REAL, "
        to_execute += "registerA REAL, "
        to_execute += "registerB REAL, "
        to_execute += "registerC REAL, "
        to_execute += "TI REAL"
        to_execute += ");"
        cursor.execute(to_execute)
    cursor.close()
    connection.close()

def chunks(data, rows = 5000):
    """
    Create chunks of data when writing data
    to the database. Significant speed up over
    adding each point individually.
    Arguments:
        data - np.array of data to be written to database
    Optional:
        rows - integer for the number of rows in a chunk
    """
    for i in range(0, len(data), rows):
        yield data[i:i+rows]

def add_to_database(table, data, database):
    """
    Write the values calculated from the pickle
    files to the database in chunks.
    Requires:
        table - string, the name of the table in the database
        data - array, data to be written to database
        database - string, name of the database file
    """
    print("Writing {}".format(table))
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    div_data = chunks(data)
    for chunk in div_data:
        cursor.execute("Begin Transaction")
        for chromophoreA, chromophoreB, distace, deltaE, alignX, alignY, alignZ, registerA, registerB, registerC, TI in chunk:
            query = "INSERT INTO {} VALUES( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);".format(table)
            cursor.execute(query, (chromophoreA, chromophoreB, distace, deltaE, alignX, alignY, alignZ, registerA, registerB, registerC, TI))
    connection.commit()
    cursor.close()
    connection.close()

def check_vector(va, vb, box):
    """
    Make sure the molecules are moved into
    the same periodic image.
    Requires:
        va - array of vector 1
        vb - array of vector 2
        box - array for the simulation box
    Returns:
        va - array in same image as vb
    """
    #Check the periodic boundaries
    temp_vec = pbc(vb-va, box)

    #Move the vector so they are in the same periodic image
    va = vb - temp_vec

    return va

def vector_along_mol(a, b, c, box):
    """
    Calculate the vector along the molecule.
    From back bond to sulfur in thiophene
    and along the long axis in DBP
    Requires:
        a, b, c - arrays for the three descriptor vectors
        box - array for the simulation box
    Returns:
        normalized array
    """
    #Check the image
    a = check_vector(a, b, box)

    #Get the average position between the two positions
    middle = np.mean([a,b], axis = 0)

    #Make sure the average is in the same vector
    #as the third
    middle = check_vector(middle, c, box)

    #Calculate the normalized vector from the averaged
    #to the third
    return (c - middle)/np.linalg.norm(c-middle)

def vector_along_back(a, b, box):
    """
    Calculate the vector across the 
    molecule.
    Requires:
        a, b - arrays for two descriptor vectors
            across the chromophore
        box - array for the simulation box
    Returns:
        normalized array
    """
    #Check the image
    a = check_vector(a, b, box)

    #get the normalized vector from a to b
    return (b - a)/np.linalg.norm(b - a)

def vector_normal(a, b, c, box):
    """
    Calculate the vector normal to
    the molecule's plane.
    Requires:
        a, b, c - arrays for the three descriptor vectors
        box - array for the simulation box
    Returns:
        normalized array normal to molecule's plane
    """
    #Make sure the a and c vectors don't cross
    #a boundary from b
    a = check_vector(a, b, box)
    c = check_vector(a, c, box)

    #Get the vecotrs from the b position to the a and c
    v1 = a - b
    v2 = c - b

    #Get the cross product of these vectors to get
    #the normal vector
    cross = np.cross(a,b)

    #Normalize the vector
    cross /= np.linalg.norm(cross)

    return cross

def generate_vectors(indices, positions, box, three_atom_indices):
    """
    Wrapper function in getting the three vectors.
    Requires:
        indices - list of the atom indices in the chromophore
        positions - array of atom positions
        box - array for the simulation box
        three_atom_indices - list, the indices of the desired
            atoms within the indices describing the chromophore
    Returns:
        v1, v2, v3 - arrays describing the 
            orientation of the chromophore
    """
    #Get the indices of the atoms within the molecule.
    i, j, k = three_atom_indices

    #Get positions for the three atoms.
    a, b, c = positions[indices[i]], positions[indices[j]], positions[indices[k]]

    #Get the three vectors
    v1 = vector_along_mol(a, b, c, box)
    v2 = vector_along_back(a, b, box)
    v3 = vector_normal(a, b, c, box)
    return v1, v2, v3

def generate_empty_dict(N):
    """
    Creates an empty dictionary for each chromophore.
    Requires:
        N - integer for number of chromophores
    Returns:
        vector_dict - empty dictionary of dictionaries.
    """
    vector_dict = {}

    #For each chromophore create an empty dictionary
    for i in range(N):
        vector_dict[i] = {}

    return vector_dict

def fill_dict(vector_dict, 
        chromophore_list, 
        AA_morphology_dict, 
        box, 
        species, 
        three_atom_indices):
    """
    Iterate through the chromophore list and
    calculate the vectors describing the chromophore
    orientations.
    Requires:
        vector_dict - empty dictionary
        chromophore_list - list of all chromophore data
        AA_morphology_dict - Dictionary of morphology data
        box - array for the simulation box
    Returns:
        vector_dict - filled dictionary where
            each key is a chromophore the pair
            is another dictionary for each
            vector.
    """
    #Get the vectors for all the chromophores of the desired species:
    for i, chromophore in enumerate(chromophore_list):
        if chromophore.species == species:
            v1, v2, v3 = generate_vectors(chromophore.CGIDs, 
                    AA_morphology_dict['position'], 
                    box, 
                    three_atom_indices)
            #Write the vectors to the dictionary.
            vector_dict[i]['vec1'] = v1
            vector_dict[i]['vec2'] = v2
            vector_dict[i]['vec3'] = v3
    return vector_dict

def run_system(table, infile, molecule_dict, species):
    """
    Calculate the relative orientational
    data for pairs of chromophores.
    Requires:
        table - string, the name of the table in the database
        infile - string, the name of the pickle file
        molecule_dict - dictionary containing molecule information
        species - string, 'donor' or 'acceptor'
    """
    #Load in the data
    AA_morphology_dict, CG_morphology_dict, CG_to_AAID_master, parameter_dict, chromophore_list = hf.load_pickle(infile)

    #Get the number of the desired species for creating the dictionary
    species_list = [chromophore.species for chromophore in chromophore_list]
    n_species = species_list.count(species)

    vector_dict = generate_empty_dict(n_species)
    #Set up the periodic simulation box into a single variable.
    box = np.array([[AA_morphology_dict['lx'], AA_morphology_dict['ly'], AA_morphology_dict['lz']]])

    vector_dict = fill_dict(vector_dict, 
            chromophore_list, 
            AA_morphology_dict, 
            box, 
            species, 
            molecule_dict['atom_indices'])

    data = []  # List for storing the calculated data

    #Because DBP has fullerenes, iterate only up to
    #where the system is DBP

    #Iterate through all the chromophores.
    for i, chromophore in enumerate(chromophore_list):
        #Only get the desired acceptor or donor species, needed for blends.
        if chromophore.species == species:
            #Iterate through the neighbors of each chromophore.
            for neighbor in zip(chromophore.neighbours, chromophore.neighbours_delta_E, chromophore.neighbours_TI):
                species1 = i
                species2 = neighbor[0][0]  # Gets the neighbor's index
                dE = neighbor[1]  # Get the difference in energy
                TI = neighbor[2]  # Get the transfer integral
                if TI > 0:  # Consider only pairs that will have hops.

                    #Get the location of the other chromophore and make sure they're in the
                    #same periodic image.
                    species2_loc = check_vector(chromophore_list[species2].posn, chromophore.posn, box)
                    #Calculate the distance (normally this is in Angstroms)
                    centers_vec = chromophore.posn - species2_loc
                    distance = np.linalg.norm(centers_vec)

                    #Get the direction from one chromophore center to the other
                    direction_vec = centers_vec/distance

                    #Get the vectors describing the two chromophores
                    vdict1 = vector_dict[species1]
                    vdict2 = vector_dict[species2]

                    #Calculate the differences in alignment between the chromophores
                    dotA = np.dot(vdict1['vec1'], vdict2['vec1'])
                    dotB = np.dot(vdict1['vec2'], vdict2['vec2'])
                    dotC = np.dot(vdict1['vec3'], vdict2['vec3'])
                    dotD = np.dot(direction_vec, vdict1['vec1'])
                    dotE = np.dot(direction_vec, vdict1['vec2'])
                    dotF = np.dot(direction_vec, vdict1['vec3'])

                    #Combine the data into an array
                    datum = np.array([species1,
                        species2,
                        distance,
                        dE,
                        dotA,
                        dotB,
                        dotC,
                        dotD,
                        dotE,
                        dotF,
                        TI])

                    data.append(datum) #Write the data to the list

    data = np.array(data) #Turn the list into an array, may be unnecessary

    add_to_database(table, data, molecule_dict['database']) #Write the data to the database

def manual_load(table, infile, C1_index, C2_index):
    """
    A manual way to load in the data when we only want
    to compare a pair of chromophores.
    Requires:
        table - string, the name of the table in the database
        infile - string, the name of the pickle file
        C1_index - integer for chromophore 1
        C2_index - integer for chromophore 2
    """
    AA_morphology_dict, CG_morphology_dict, CG_to_AAID_master, parameter_dict, chromophore_list = hf.load_pickle(infile)
    box = np.array([[AA_morphology_dict['lx'], AA_morphology_dict['ly'], AA_morphology_dict['lz']]])
    write_positions_to_xyz(C1_index, C2_index, chromophore_list, AA_morphology_dict, box)

def write_positions_to_xyz(C1_index, C2_index, chromophore_list, AA_morphology_dict, box):
    """
    Write the atom positions in two chromophores
    to an xyz file so it can be viewed
    Requires:
        C1_index - integer for chromophore 1
        C2_index - integer for chromophore 2
        chromophore_list - list of all chromophore data
        AA_morphology_dict - Dictionary of morphology data
        box - array for the simulation box
    """

    #Name the file based on the two chromophores of interest.
    filename = "{}-{}.xyz".format(C1_index, C2_index)

    #Get the first's positions and types
    chromophoreA = chromophore_list[C1_index]
    chromophoreA_positions = [AA_morphology_dict['position'][i] for i in chromophoreA.AAIDs]
    chromophoreA_types = [AA_morphology_dict['type'][i] for i in chromophoreA.AAIDs]

    #Get the second's positions and types
    chromophoreB = chromophore_list[C2_index]
    chromophoreB_positions = [AA_morphology_dict['position'][i] for i in chromophoreB.AAIDs]
    chromophoreB_types = [AA_morphology_dict['type'][i] for i in chromophoreB.AAIDs]

    #Combine two chromophores into one list
    types = chromophoreA_types + chromophoreB_types
    positions = chromophoreA_positions + chromophoreB_positions
    positions = np.array(positions)

    #Create a reference positions so that all others will be placed into
    #the same image as the reference position
    reference = np.copy(positions[0])

    #Move the atoms to the same reference position
    for i, pos in enumerate(positions):
        saved = np.copy(pos)
        positions[i] = check_vector(pos, reference, box) 

    #Center the chromophores around the geometric average
    positions -= np.mean(positions, axis = 0)

    #Get the number of atoms in the two chromophores
    n_atoms = len(positions)

    #Write the data to the xyz file
    to_write = "{}\n\n".format(n_atoms)
    for i, atom in enumerate(positions):
        atom_type = types[i]
        to_write += "{} {} {} {}\n".format(atom_type, atom[0], atom[1], atom[2])
    with open(filename, 'w') as f:
        f.write(to_write)
        
def create_systems(subdir):
    """
    Creates a dictionary with table:
    pickle file pairs. 
    The name before the .pickle is used
    as the table name.
    The name of the pickle file must be
    sql friendly.
    Requires:
        subdir - string for the directory
            of pickle files
    Returns:
        systems - dictionary
    """
    systems = {}

    #Get all the pickle files
    directory_list = glob(subdir)

    #From the pickle file names, convert them
    #into an sql friendly format and write
    #them to the dictionary.
    for directory in directory_list:
        name = directory.split('/')[-1]
        name = name.split(".")[0]
        systems[name] = directory
    return systems

def get_molecule_dictionary(mol):
    """
    Function contains the dictionary
    of molecule dictionaries.
    This should centralize the hard-coding
    for future molecules.
    Currently supported molecules are p3ht and dbp.
    Requires:
        mol - string, molecule name
    Returns:
        molecule dictionary - dict
    """
    molecules = {'p3ht':{'database':'p3ht.db',
        'subdir':'training_data/P3HT/*.pickle',
        'atom_indices':[0, 1, 3]},
            'dbp':{'database':'dbp.db',
                'subdir':'training_data/DBP/*.pickle',
                'atom_indices':[0, 20, 38]}}
    return molecules[mol]

def rename_DBP_systems_to_sql_friendly():
    """
    Helper function to rename the DBP
    pickle files to be an sql friendly naming
    scheme so that the sql tables can use the
    file names.
    Requires:
        None
    Returns:
        None
    """
    directory = "training_data/DBP"
    pickle_files = glob(directory+'/*.pickle')

    for pickle in pickle_files:
        name = pickle.split('/')[-1]
        assert len(name.split('.')) > 2 
        #Only allow the rename if there are too many periods.
        name = name.split("-")
        mol = name[1]
        temps = name[-1].split(".")
        name = "DBP_{}_{}_{}.pickle".format(mol, temps[0], temps[1])
        os.rename(pickle, os.path.join(directory, name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--species", required=False,
            help="""Pass this flag so that only pairs
            of the specified species (donor or acceptor)
            are added into the database.
            If flag is not passed, defaults to 'donor'.
            """)
    args, input_list = parser.parse_known_args()

    species = 'donor'
    if args.species:
        species = args.species

    molecule = sys.argv[1]

    molecule_dict = get_molecule_dictionary(molecule)

    systems = create_systems(molecule_dict['subdir'])

    create_data_base(molecule_dict['database'], systems)

    for key, pair in systems.items():
        run_system(key, pair, molecule_dict, species)

if __name__ == "__main__":
    main()
    #manual_load('crystalline', systems['crystalline'], 0, 1)
