import pickle
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---------------- USER INPUT ----------------
"""
  Define the number of clusters (i.e., the number of structures to select) for the k-means algorithm and the number of files to generate with the selected structures.
  Note: If you select a small number of structures, e.g., 5, each file will likely contain the same 5 structures. For a larger number of structures,
  the selected structures may differ, as the initial positions of the centroids in k-means are determined randomly.
"""
num_clusters = 30
how_many_files_to_generate = 50

"""
 Possibly change weights for the individual parts of the total fingerprint. CrystalNN has 244 elements, 
 then 4 elements for difference in local atomic masses, 4 elements for average bond distances, and 100 elements for RDF up to 10 A 
"""
weight_crystalNN = 1
weight_element_mass_difference = 4
weight_average_bond_distnace = 2
weight_RDF = 1

# List of files containing structure names and their fingerprints - dictionaries generated by the script 'Fingerprint_calculation.py'
list_files = ['dict_STD_0_1.pkl', 'dict_STD_0_2.pkl','dict_STD_0_5.pkl',
              'dict_VACA_0_1.pkl','dict_VACA_0_2.pkl', 'dict_VACA_0_5.pkl']

# Descriptive names for the individual dictioners (will be used in plots for t-SNE)
list_of_files_names = ['STD_0.1', 'STD_0.2', 'VACA_0_1', 'VACA_0_2']


# ------------------- LOADING DATA -------------------
database = {}
size_of_groups = []
import re
regex = r"_([0-9]*\.[0-9]+)_" #this matches numbers in the names of files, e.g. _123.456_, or _.75_, in case
#regex = r"(\d+\.\d+)%"



markers = ['o', 's', '^', 'o', 's', '^', 'o', 's', '^']
map_type = ['Greens', 'Blues', 'Reds', 'Greens', 'Blues', 'Reds', 'Greens', 'Blues', 'Reds']

color_map = []

# Load data from each file in 'list_files' list and update the database
for file_name in list_files:
    with open(file_name, 'rb') as input_file:
        size_before = len(database)
        database.update(pickle.load(input_file))
        size_after = len(database)
        elements_added = size_after - size_before
        size_of_groups.append(elements_added)
        # Extracting the numerical values


        input_file.seek(0)
        data = pickle.load(input_file)

        #extracted_values = [float(re.search(regex, key).group(1)) for key in data.keys()]
        extracted_values = [abs(float(key.split('_')[-1].rstrip('%'))) for key in data.keys()]
        color_map.append(extracted_values)
    print(f"Data from {file_name} loaded successfully.")


"""
def remove_keywords_from_dict(dict_to_modify, file_with_keywords):
    #Remove keywords and their corresponding values from a dictionary based on a list of keywords provided in a file.
    try:
        with open(file_with_keywords, 'r') as file:
            keywords = file.readlines()
            # Strip whitespace and newline characters from each keyword
            keywords = [keyword.strip() for keyword in keywords]

        # Remove each keyword from the dictionary
        for keyword in keywords:
            if keyword in dict_to_modify:
                del dict_to_modify[keyword]

    except FileNotFoundError:
        print(f"The file {file_with_keywords} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
keywords_file = 'files_with_large_vectors.txt'  # Replace 'keywords.txt' with your file's name
remove_keywords_from_dict(database, keywords_file)
"""

def extract_number(key):
    parts = key.split('_')
    return int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0

# Extract names and fingerprints
names = list(database.keys())
fingerprints = np.array(list(database.values()))
print("Length of the first loaded fingerprint:       {}".format(len(fingerprints[1])))
print(f"Total number of loaded structures:       {len(names)}")
#----------------------------------------------------------------


ikx = 0
while ikx < how_many_files_to_generate:
    print('\n--------------------------------------------------------')
    print("Started to preparing file number {}/{} with total {} atomic configurations selected.".format(ikx,how_many_files_to_generate, num_clusters))
    names = list(database.keys())
    fingerprints = np.array(list(database.values()))
    scaler = StandardScaler()
    fingerprints = scaler.fit_transform(fingerprints)

    # Setting the weights on individual fingerprint parts:
    fingerprints[:, :244] *= weight_crystalNN
    fingerprints[:, 244:248] *= weight_element_mass_difference
    fingerprints[:, 248:252] *= weight_average_bond_distnace
    fingerprints[:, 252:] *= weight_RDF

    #---------------- PRINCIPAL COMPONENT ANALYSIS (PCA) ----------------
    #Choosing reasonable number of PCA component according to the cumulative explained variance up to 98 %
    pca = PCA()
    pca.fit(fingerprints)
    explained_variance_ratio = pca.explained_variance_ratio_
    principal_components = pca.fit_transform(fingerprints)
    fingerprints=principal_components
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    index_greater = np.where(cumulative_explained_variance >0.98)[0]
    print("Cumulative explained variance is larger than 0.98 above the PCA component of: {}\n "
          "Choosing the number of components equal to this number.".format(index_greater[0]))
    #Performing PCA reduction
    print("Performing PCA reduction.")
    pca = PCA(n_components=index_greater[0])
    principal_components = pca.fit_transform(fingerprints)
    fingerprints=principal_components
    from mpl_toolkits.mplot3d import Axes3D
    print("PCA done.")
    #----------------------------------------------------------------



    #---------------- K-MEANS CLUSTERING ----------------
    print("Performing K-means clustering on the reduced PCA components.")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, tol=1e-6,
                    algorithm='lloyd',max_iter=600, n_init=num_clusters-round(0.5*num_clusters)).fit(fingerprints)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_


  
    # ---------------- If you wish the find indecis of vectors which are farthers from each other instead of only selecting vectors which are closest to the centroids
    """
    cluster_points = {i: [] for i in range(len(centroids))}
    cluster_indices = {i: [] for i in range(len(centroids))}
    
    for idx, cluster in enumerate(labels):
        cluster_points[cluster].append(fingerprints[idx])  
        cluster_indices[cluster].append(idx) 
    for i in range(len(centroids)):
        cluster_points[i] = np.array(cluster_points[i])
        cluster_indices[i] = np.array(cluster_indices[i])
    
    # Step 3: Find the farthest point from each centroid
    selected_indices = []
    
    for cluster_id in range(len(centroids)):
        points = cluster_points[cluster_id]
        indices = cluster_indices[cluster_id]
    
        # Find the farthest point from the centroid
        farthest_idx = np.argmax([euclidean(point, centroids[cluster_id]) for point in points])
        selected_indices.append(indices[farthest_idx])  # Store the original index
    
    # Step 4: Iterate over possible selections to maximize distance
    best_combination = selected_indices
    max_distance = sum(euclidean(fingerprints[i], fingerprints[j]) for i, j in combinations(selected_indices, 2))
    
    # Step 5: Try alternative selections for maximal distance
    for candidates in zip(*cluster_indices.values()):  # Iterate over possible selections
        total_distance = sum(euclidean(fingerprints[i], fingerprints[j]) for i, j in combinations(candidates, 2))
        if total_distance > max_distance:
            best_combination = candidates
            max_distance = total_distance
    
    for vall in best_combination:
        print(vall)
        selected_structure_name = names[vall]
        print(selected_structure_name)
        selected_structures[selected_structure_name] = database[selected_structure_name]
        selected_fingerprints.append(fingerprints[vall])
    
    #Uncomment the section above and comment the section below if you wish to find farthers vectors (one from each cluster) instead of selecting only vectors that are each closest to the centroid.
    #Do not forget to also uncomment corresponding part in the plotting of tnse (ax2.scatter(tsne_results[best_combination, 0], tsne_results[best_combination, 1], s=300,color='black', alpha=0.5))
    for vall in best_combination:
        print(vall)
        selected_structure_name = names[vall]
        print(selected_structure_name)
        selected_structures[selected_structure_name] = database[selected_structure_name]
        selected_fingerprints.append(fingerprints[vall])

#Uncomment the section above and comment the section below if you wish to find farthers vectors (one from each cluster) instead of selecting only vectors that are each closest to the centroid.
"""  
    # Select structures closest to each cluster centroid
    selected_structures = {}
    selected_fingerprints = []
    indecis_added = []
    for cluster in range(num_clusters):
        indices = np.where(labels == cluster)[0]
        other_indices = np.where(labels != cluster)[0]
        if len(indices) > 0:
            closest_point_index = indices[np.argmin([euclidean(fingerprints[idx], centroids[cluster]) for idx in indices])]
            indecis_added.append(closest_point_index)
            selected_structure_name = names[closest_point_index]
            selected_structures[selected_structure_name] = database[selected_structure_name]
            selected_fingerprints.append(fingerprints[closest_point_index])
    #---------------------------------------------------------------------------



      
        # Write sorted names of selected structures to a file
        sorted_structure_names = sorted(selected_structures, key=extract_number)
        with open('selected_structure_names_PCA_{}.txt'.format(ikx), 'w') as file:
            file.writelines(name + '.xml\n' for name in sorted_structure_names)
    print('Done\n-------------------------------------------------------')



    """
     #The following is for testing purposes
    print(sorted(indecis_added))
    all_indecis = range(0, len(names))
    indecis_not_added = [item for item in all_indecis if item not in indecis_added]
    print(indecis_not_added)
    dist = [sum([euclidean(fingerprints[idxx], fingerprints[idx]) for idx in indecis_added]) for idxx in indecis_not_added]
    threshold = 0.0001  # Define a threshold for almost zero
    # Calculate distances and filter out the ones that are almost zero
    # Filter out indices from indecis_not_added where all distances are almost zero. This is useful when I want to filter out structure is essentially the same as other one,
    #maybe because it is supercell, but differs most from the others.
    indecis_not_added = [idxx for idxx in indecis_not_added if all(euclidean(fingerprints[idxx], fingerprints[idx]) > threshold for idx in indecis_added)]
    dist = [sum([euclidean(fingerprints[idxx], fingerprints[idx]) for idx in indecis_added]) for idxx in indecis_not_added]
    sorted_pairs = sorted(zip(indecis_not_added, dist), key=lambda pair: pair[1], reverse=True)
    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_pairs]
    sorted_values = [value for index, value in sorted_pairs]
    other_names_possibly_add = [names[i] for i in sorted_indices]
    with open('Add_structures_to_previous_{}.txt'.format(num_clusters), 'w') as f0:
        f0.write("#Structure\tDistance\n")
        for value1, value2 in zip(other_names_possibly_add, sorted_values):
            f0.write(f'{value1}\t{value2}\n')
    """
    ikx = ikx + 1
