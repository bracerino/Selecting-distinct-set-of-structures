#Name of this script: Fingerprint_Calculator.py
import os
import numpy as np
import pickle
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.local_env import VoronoiNN, NearNeighbors
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.site.bonding import AverageBondLength
from matminer.featurizers.site.rdf import GeneralizedRadialDistributionFunction
from pymatgen.analysis.local_env import MinimumDistanceNN

class StructureFingerprintCalculator:
    def __init__(self, directory_path, output_dict_file, chem_info):
        self.directory_path = directory_path
        self.output_dict_file = output_dict_file
        self.chem_info = chem_info
        self.opp = {1: ['wt',
        'sgl_bd'],
    2: ['wt',
        'L-shaped',
        'water-like',
        'bent '
        '120 '
        'degrees',
        'bent '
        '150 '
        'degrees',
        'linear'],
    3: ['wt',
        'trigonal '
        'planar',
        'trigonal '
        'non-coplanar',
        'T-shaped'],
    4: ['wt',
        'square '
        'co-planar',
        'tetrahedral',
        'rectangular '
        'see-saw-like',
        'see-saw-like',
        'trigonal '
        'pyramidal'],
    5:   ['wt',
        'pentagonal '
        'planar',
        'square '
        'pyramidal',
          'trigonal '
          'bipyramidal'
          ],
    6: ['wt',
        'hexagonal '
        'planar',
        'octahedral',
        'pentagonal '
        'pyramidal'],
    7: ['wt',
       'hexagonal '
       'pyramidal',
       'pentagonal '
       'bipyramidal'],
    8: ['wt',
        'body-centered '
        'cubic',
        'hexagonal '
        'bipyramidal'],
    9: ['wt',
        'q2',
        'q4',
        'q6'],
    10: ['wt',
         'q2',
         'q4',
         'q6'],
    11: ['wt',
         'q2',
         'q4',
         'q6'],
    12: ['wt',
         'cuboctahedral',
         'q2',
         'q4',
         'q6'],
    13: ['wt'],
    14: ['wt'],
    15: ['wt'],
    16: ['wt'],
    17: ['wt'],
    18: ['wt'],
    19: ['wt'],
    20: ['wt'],
    21: ['wt'],
    22: ['wt'],
    23: ['wt'],
    24: ['wt']}

        self.cnn = {1: ['wt'],
    2: ['wt'],
    3: ['wt'],
    4: ['wt'],
    5: ['wt'],
    6: ['wt'],
    7: ['wt'],
    8: ['wt'],
    9: ['wt'],
    10: ['wt'],
    11: ['wt'],
    12: ['wt'],
    13: ['wt'],
    14: ['wt'],
    15: ['wt'],
    16: ['wt'],
    17: ['wt'],
    18: ['wt'],
    19: ['wt'],
    20: ['wt'],
    21: ['wt'],
    22: ['wt'],
    23: ['wt'],
    24: ['wt']}
        # Copy the 'cnn' dictionary here
        self.nn_method = VoronoiNN()
        self.structure_features = {}
        #Will calculate fingerprint for each atomic position and then average all corresponding elements from them,
        # creating 'mean', 'std_dev', 'minimum', 'maximum' for each part of CN_x
        #Creating instances for CrystalNN, average crystal bond length, and RDF
        #distance_cutoffs = [(0, 20), (5, 20)]
        self.ssf = SiteStatsFingerprint(
            CrystalNNFingerprint(op_types=self.opp, chem_info=self.chem_info, distance_cutoffs=None, x_diff_weight=0),
            stats=('mean', 'std_dev', 'minimum', 'maximum'))
        #print(self.ssf.implementors())
        self.abl_featurizer = AverageBondLength(self.nn_method)
        self.GRDF = GeneralizedRadialDistributionFunction.from_preset('gaussian')
        self.GRDF = GeneralizedRadialDistributionFunction.from_preset('gaussian', width=0.1, spacing=0.1, cutoff=10, mode='GRDF')
        #print(self.GRDF.feature_labels())
        #print(self.GRDF)
    def read_structure_from_POSCAR(self, file_path):
        print(f'--------------------------------------------')
        print(f"Reading structure from: {file_path}")
        poscar = Poscar.from_file(file_path)
        return poscar.structure

    def calculate_features(self, structure):
        # ssf = SiteStatsFingerprint(
        #    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
        #    stats=('mean', 'std_dev', 'minimum', 'maximum'))
        # When using .from_present, it is not possible to add chem_info. Need to define all the parameters seperately, thus the definition of all
        # elements in 'opp' above
        print("Calcuting its CrystalNN fingerprint...")
        # Calcuting fingerprint (accounting for the difference in local atomic masses) for the whole structure = vector of 248 elements
        try:
            fingers = np.array(self.ssf.featurize(structure))
        except Exception as e:
            print("It is not possible to calculate CrystalNN for this structure. There is probably a vacuum. The initial 244 elements will be set to zero.\n"
                  "Differences in atomic masses will still be calculated.")
            fingers0 = np.zeros(244)
            # Initialize MinimumDistanceNN with a specified maximum cutoff distance
            cutoff_distance = 5.0  # Set a suitable cutoff distance in Angstroms
            mdnn = MinimumDistanceNN(cutoff=cutoff_distance)
            # Calculate local differences in atomic mass
            atomic_mass_differences = []

            for i, site in enumerate(structure):
                neighbors = mdnn.get_nn_info(structure, i)
                central_mass = site.specie.atomic_mass
                local_diffs = [central_mass - neighbor['site'].specie.atomic_mass for neighbor in neighbors]
                average = np.mean(local_diffs)
                atomic_mass_differences.append(average)
            average = np.mean(atomic_mass_differences)
            std_dev = np.std(atomic_mass_differences)
            min_value = np.amin(atomic_mass_differences)
            max_value = np.amax(atomic_mass_differences)
            to_append = np.array([average, std_dev, min_value, max_value])
            fingers=np.append(fingers0, to_append)
        #Calculating average bond length for the whole structure, adding 4 elements (mean, std, minimum, maximum of bond length
        # to the previous vector of 248 ---> 252
        bond_lengths = []
        print("Calcuting its average BOND length within the structure...")
        try:
            for idx in range(len(structure)):
                val = self.abl_featurizer.featurize(structure, idx)[0]
                bond_lengths.append(val)
            bond_lengths = np.array(bond_lengths)
            average = np.mean(bond_lengths)
            std_dev = np.std(bond_lengths)
            min_value = np.amin(bond_lengths)
            max_value = np.amax(bond_lengths)

            to_append = np.array([average, std_dev, min_value, max_value])
            fingerprint_output_0 = np.append(fingers, to_append)
        except Exception as e:
            cutoff_distance = 5.0  # Set a suitable cutoff distance in Angstroms
            self.nn_method = MinimumDistanceNN(cutoff=cutoff_distance)
            self.abl_featurizer = AverageBondLength(self.nn_method)
            for idx in range(len(structure)):
                val = self.abl_featurizer.featurize(structure, idx)[0]
                bond_lengths.append(val)
            bond_lengths = np.array(bond_lengths)
            average = np.mean(bond_lengths)
            std_dev = np.std(bond_lengths)
            min_value = np.amin(bond_lengths)
            max_value = np.amax(bond_lengths)
            to_append = np.array([average, std_dev, min_value, max_value])
            fingerprint_output_0 = np.append(fingers, to_append)

        print("Calcuting its PAIR DISTRIBUTION FUNCTION...")
        bond_lengths = []
        #Calcuting PDF for each atomic site, then averaging all of them. Adding 10 values of the RDF to the
        #characteristic vector of the structure ---> 262 elements
        for idx in range(len(structure)):
            cc = self.GRDF.featurize(structure, idx)
            bond_lengths.append(cc)
        bond_lengths = np.array(bond_lengths)
        average = np.mean(bond_lengths, axis=0)
        fingerprint_output = np.append(fingerprint_output_0, average)
        #fingerprint_output = fingerprint_output_0
        # print(fingerprint_output)
        print("Calculation successful.")
        print(f'--------------------------------------------')
        return fingerprint_output

    def calculate_difference(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def process_structures(self):
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            structure = self.read_structure_from_POSCAR(file_path)
            self.structure_features[filename] = self.calculate_features(structure)

        file_differences = {}
        with open(self.output_dict_file, 'wb') as f:
            pickle.dump(self.structure_features, f)
        print("The structure's fingerprints were saved in a dictionary: {}".format(self.output_dict_file))

        for filename1, features1 in self.structure_features.items():
            #print(f"Comparing {filename1} with other structures...")
            differences = []
            for filename2, features2 in self.structure_features.items():
                if filename2 != filename1:
                    diff = self.calculate_difference(features1, features2)
                    differences.append((filename2, diff))
            differences.sort(key=lambda x: x[1])
            file_differences[filename1] = differences[:10]
        #print(f"Top 10 lowest differences for {filename1} calculated.")

        self.write_results(file_differences)

    def write_results(self, file_differences):
        output_file = 'structure_differences.txt'
        with open(output_file, 'w') as f:
            for file, diffs in file_differences.items():
                f.write(f"{file}: " + ', '.join([f"{d[0]} (= {d[1]:.2f})" for d in diffs]) + '\n')
        print("Processing complete. \n"
              "Now use either Kmeans or DBSCAN clustering script to select the most different structures\n"
              "from this dictionary.")


''' Usage
chem_info2 = {
    "atomic_mass": {
        "Ti": 47.867,
        "Al": 26.98,
        "Cr": 51.9961,
        "Fe": 55.845,
        "Ni": 58.6934,
        "Cu": 63.546
    }
}
calculator = StructureFingerprintCalculator('POSCAR_surf/', 'dict_fingerprints_diffsssssss.pkl', chem_info2)
calculator.process_structures()
#'''
