import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# ---------------- USER INPUT ----------------
# Set the number of clusters (number of structures to be selected) for k-means algorithm
num_clusters = 30

# List of files containing
list_files = ['dict_STD.pkl', 'dict_VACA.pkl'] #'INT_0_1',' INT_0_2', 'INT_0_5',
list_files = ['dict_STD_0_1.pkl', 'dict_STD_0_2.pkl','dict_STD_0_5.pkl',
              'dict_VACA_0_1.pkl','dict_VACA_0_2.pkl', 'dict_VACA_0_5.pkl']
#list_files = ['dict_STD_2.pkl', 'dict_INT_2.pkl', 'dict_VACA_2.pkl']

list_of_files_names = [
                       'Standard 0.1', 'Standard 0.2', 'Standard 0.5',
                       'Vaca 0.1', 'Vaca 0.2', 'Vaca 0.5']
#'INT_0_1',' INT_0_2', 'INT_0_5',
#list_of_files_names = ['STD', 'INT', 'VACA']


#------------------- LOADING DATA -------------------
database = {}
size_of_groups = []
import re
regex = r"_([0-9]*\.[0-9]+)_"
#regex = r"(\d+\.\d+)%"


color_map = []
markers = ['o', 's', '^', 'o', 's', '^', 'o', 's', '^', 'o', 's', '^']
markers = ['o', 'o', 'o', '^', '^', '^', 'o', 's', '^', 'o', 's', '^', 'o', 's', '^']
map_type = ['Greens', 'Reds', 'Blues',  'Purples', 'Oranges', 'pink', 'Greens', 'Blues', 'Reds', 'Greens', 'Blues', 'Reds']

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

        #extracted_values = [abs(float(key.split('_')[-1].rstrip('%'))) for key in data.keys()]
        extracted_values = [float(key.split('_')[-1].rstrip('%')) for key in data.keys()]
        color_map.append(extracted_values)
        print(extracted_values)
    print(f"Data from {file_name} loaded successfully.")

print(size_of_groups)
def extract_number(key):
    parts = key.split('_')
    return int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0


# Extract names and fingerprints
names = list(database.keys())
fingerprints = np.array(list(database.values()))
print(len(fingerprints[0]))
print(f"Total number of loaded structures: {len(names)}")
#------------------------------------------------------



# ---------------- Transforming the initial data ----------------
scaler = StandardScaler()
fingerprints = scaler.fit_transform(fingerprints)
#Setting the weights on individual fingerprint parts:
#Crystalnn has 244 elements, then 4 elements for atomic masses, 4 elements for average bond distance, 100 elements for RDF up to 10 A cutoff
weight_crystalNN = 1
weight_element_mass_difference = 4
weight_average_bond_distnace = 2
weight_RDF = 1
fingerprints[:, :244] *= weight_crystalNN
fingerprints[:, 244:248] *= weight_element_mass_difference
fingerprints[:, 248:252] *= weight_average_bond_distnace
fingerprints[:, 252:] *= weight_RDF
print("LLLLLLLL")
print(fingerprints[:, :244].shape[1] )
print(fingerprints[:, 244:248].shape[1] )
print(fingerprints[:, 248:252].shape[1] )
print(fingerprints[:, 253:].shape[1] )
print(fingerprints[0] )
#----------------------------------------------------------------



# ---------------- Saving mean and variance, only for testing purposes ----------------
means = np.mean(fingerprints, axis=0)
# Calculate the variance for each element across all vectors
variances = np.var(fingerprints, axis=0)
with open('means_variance_vectors.txt', 'w') as f0:
    for idx, val in enumerate(means):
        f0.write("{}\t{}\n".format(means[idx], variances[idx]))
#----------------------------------------------------------------



#------------------------ PCA ------------------------
#Choosing reasonable number of PCA component according to the cumulative explained variance
pca = PCA()
#pca.fit(fingerprints[:, 0:244])
pca.fit(fingerprints)
explained_variance_ratio = pca.explained_variance_ratio_
principal_components = pca.fit_transform(fingerprints)
fingerprints=principal_components
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
index_greater = np.where(cumulative_explained_variance >0.98)[0]
print("Cumulative explained variance is larger than 0.98 above the PCA component of: {}\n "
      "Choosing number of components equal to this.".format(index_greater[0]))



import pandas as pd

loadings = pca.components_.T
print(loadings)
# Square the loadings to get the variance contribution from each feature
squared_loadings = loadings ** 2

# Select only the first 85 components
squared_loadings_first_85 = squared_loadings[:, :85]

# Sum the contributions across the first 85 components for each feature
contribution_vector_85 = squared_loadings_first_85.sum(axis=1)

# Create a DataFrame for better visualization (optional)
contribution_df = pd.DataFrame({
    'Feature': [f'Feature{i+1}' for i in range(len(contribution_vector_85))],
    'Contribution': contribution_vector_85
})




# Define the indices for the different parts
part1_indices = range(0, 244)    # First 242 elements
part2_indices = range(244, 248)  # Next 4 elements
part3_indices = range(248, 252)  # Next 4 elements
part4_indices = range(252, 352)  # Last 100 elements

# Create the plot
plt.figure(figsize=(16, 12))

# Plot each segment with a different color
plt.scatter(part1_indices, contribution_vector_85[part1_indices], marker='o', linestyle='-', color='b', label='CrystalNN\n(0-243)')
plt.scatter(part2_indices, contribution_vector_85[part2_indices], marker='o', linestyle='-', color='g', label='Atomic mass difference\n(244-247)')
plt.scatter(part3_indices, contribution_vector_85[part3_indices], marker='o', linestyle='-', color='r', label='Pair bond distances\n(248-251)')
plt.scatter(part4_indices, contribution_vector_85[part4_indices], marker='o', linestyle='-', color='m', label='RDF\n(252-351)')

# Add labels and title
plt.xlabel('Feature Index', fontsize=25)
plt.ylabel('Contributions to 98% of Explained Variance', fontsize=25)
plt.title('Feature Contribution to the First 85 PCA Components', fontsize=25)

# Add a legend to distinguish parts
plt.legend(fontsize = 16)

# Show grid for better readability
#plt.grid(True)


plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Display the plot
plt.show()







#Performing PCA reduction
print("Performing PCA reduction.")
pca = PCA(n_components=index_greater[0])
principal_components = pca.fit_transform(fingerprints)
fingerprints=principal_components
from mpl_toolkits.mplot3d import Axes3D
print("PCA done.")
#----------------------------------------------------------------






# Plot cumulative explained variance with increased sizes
plt.figure(figsize=(16, 12))  # Increase figure size
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='x', linestyle='none', color='b', markersize=8, linewidth=3)
plt.axhline(y=0.98, color='g', linestyle='-', linewidth=3)
plt.axvline(x=index_greater[0] + 1, color='g', linestyle='-', linewidth=3)

# Add annotations for the red lines
#plt.annotate('Threshold: 0.98', xy=(0.5, 0.98), xytext=(1, 0.85), textcoords='axes fraction', arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16, color='red')
#plt.annotate(f'Component: {index_greater[0] + 1}', xy=(index_greater[0] + 1, 0.5), xytext=(index_greater[0] + 1.5, 0.6), textcoords='data', arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16, color='red')

# Increase title and labels sizes
plt.xlabel('Number of Principal Components', fontsize=25)  # Increase x-label size
plt.ylabel('Cumulative Explained Variance', fontsize=25)  # Increase y-label size

# Increase x-ticks and y-ticks sizes
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.grid(True)
plt.show()



#------------------------ Performing K-means clustering ------------------------
print("Performing K-means clustering on principal components.")
kmeans = KMeans(n_clusters=num_clusters, random_state=0, tol=1e-6,
                algorithm='lloyd',max_iter=600, n_init=num_clusters-round(0.5*num_clusters)).fit(fingerprints)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
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






# Calculate distances between selected fingerprints
#selected_fingerprints = np.array(selected_fingerprints)
# Calculate Silhouette Score and Davies-Bouldin Index
print(f"Silhouette Score: {silhouette_score(fingerprints, labels)}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(fingerprints, labels)}")

# Write sorted names of selected structures to a file
sorted_structure_names = sorted(selected_structures, key=extract_number)
with open('selected_structure_names_PCA.txt', 'w') as file:
    file.writelines(name + '\n' for name in sorted_structure_names)
print("Done.")



#------------------ TSNE 2D ------------------------
fig2, ax2 = plt.subplots()
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
tsne_results = tsne.fit_transform(principal_components)
# Plotting the results
previous_val = 0
for idx, val in enumerate(size_of_groups):
    indecis_to_plot=val+previous_val
    indecis = range(previous_val, indecis_to_plot)
    previous_val=indecis_to_plot
    print(indecis_to_plot)
    #if list_of_files_names[idx] == 'STABLE':
    if idx == 0:
        #ax2.scatter(tsne_results[indecis, 0], tsne_results[indecis, 1], label=list_of_files_names[idx], s = 500)
        #sc = ax2.scatter(tsne_results[indecis, 0], tsne_results[indecis, 1], label=list_of_files_names[idx], s=500, c = color_map[idx], cmap = map_type[idx])

        sc = ax2.scatter(tsne_results[indecis, 0], tsne_results[indecis, 1], label=list_of_files_names[idx], alpha=1,
                         s=100, c=color_map[idx], cmap=map_type[idx], marker=markers[idx])
        cbar = plt.colorbar(sc, ax=ax2)

    else:
      #  sc = ax2.scatter(tsne_results[indecis, 0], tsne_results[indecis, 1],label=list_of_files_names[idx], alpha=1, s = 100,
      #              c=color_map[idx], cmap=map_type[idx], edgecolor='k', marker = markers[idx])
        sc = ax2.scatter(tsne_results[indecis, 0], tsne_results[indecis, 1], label=list_of_files_names[idx], alpha=1,
                         s=100,c=color_map[idx], cmap=map_type[idx], marker=markers[idx])
                   # c = color_map[idx], cmap = map_type[idx], edgecolor='k', marker = markers[idx])



cbar.set_label('Color Scale of % lattice deformation', fontsize=25)
cbar.ax.tick_params(labelsize=20)

ax2.set_title('2D t-SNE visualization of PCA-transformed data', fontsize = 25)
ax2.set_xlabel('t-SNE feature 1', fontsize = 25)
ax2.set_ylabel('t-SNE feature 2', fontsize = 25)

plt.xticks(fontsize=25)
ax2.legend(fontsize=19)
plt.yticks(fontsize=25)

for cluster in range(num_clusters):
    indices = np.where(labels == cluster)[0]
    if len(indices) > 0:
        closest_point_index = indices[np.argmin([euclidean(fingerprints[idx], centroids[cluster]) for idx in indices])]
        ax2.scatter(tsne_results[closest_point_index, 0], tsne_results[closest_point_index, 1],
                   s=300,color='black', alpha=0.5)
import random
random_numbers = random.sample(range(2700), num_clusters)
#ax2.scatter(tsne_results[random_numbers, 0], tsne_results[random_numbers, 1],
#            s=300, color='purple', alpha=0.5)
sorted_values = [value for index, value in sorted_pairs]
#other_names_possibly_add = [names[i] for i in random_numbers]
#with open('Selected_RANDOM.txt'.format(num_clusters), 'w') as f0:
#    for value1 in other_names_possibly_add:
#        f0.write(f'{value1}\n')


# Plotting the results in 3D
plt.figure(figsize=(10, 8))
ax = plt.figure().add_subplot(projection='3d')

previous_val = 0


tsne = TSNE(n_components=3, random_state=0, perplexity=30)
tsne_results = tsne.fit_transform(principal_components)
for idx, val in enumerate(size_of_groups):
    indecis_to_plot=val+previous_val
    indecis = range(previous_val, indecis_to_plot)
    previous_val=indecis_to_plot
    print(indecis_to_plot)
   # ax.scatter(tsne_results[indecis, 0], tsne_results[indecis, 1], tsne_results[indecis, 2], label=list_of_files_names[idx])
#ax2.legend()

ax.legend()
ax.set_title('3D t-SNE visualization of PCA-transformed data')
ax.set_xlabel('t-SNE feature 1')
ax.set_ylabel('t-SNE feature 2')
ax.set_zlabel('t-SNE feature 3')


for cluster in range(num_clusters):
    indices = np.where(labels == cluster)[0]
   # if len(indices) > 0:
      #  closest_point_index = indices[np.argmin([euclidean(fingerprints[idx], centroids[cluster]) for idx in indices])]
      #  ax.scatter(tsne_results[closest_point_index, 0], tsne_results[closest_point_index, 1], tsne_results[closest_point_index, 2],
      #             s=300,color='black', alpha=0.8
      #             )



plt.show()



# Number of components
num_components = len(explained_variance_ratio)


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_components + 1), cumulative_explained_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components',  fontsize=25)
plt.ylabel('Cumulative Explained Variance',  fontsize=25)
tick_interval = 25  # Set the interval for ticks (e.g., every 5 components)
plt.xticks(range(1, num_components + 1, tick_interval))
plt.grid(True)
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Principal Component {i + 1}: {ratio:.2f}")

# Increase title and labels sizes

# Increase x-ticks and y-ticks sizes
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

#@transposed_array = vectors_array.T

# Calculate the correlation matrix
#corr_matrix = np.corrcoef(transposed_array)
#scaler = MinMaxScaler(feature_range=(-1, 1))


#scaler.fit(vectors_array)
# Transform the data
#standardized_vectors = scaler.transform(vectors_array)
#vectors_array=standardized_vectors
print("VEC")
#vectors_array[:, 244:248] *= 50
pca = PCA()
#pca.fit(vectors_array)
#print(vectors_array)
# Explained variance ratio
#explained_variance_ratio = pca.explained_variance_ratio_
#cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Number of components
#num_components = len(explained_variance_ratio)
#


# Show the plot
plt.show()
