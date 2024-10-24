# Import necessary libraries
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import re
import pandas as pd

# ---------------- USER INPUT ----------------
# Set the number of clusters for the k-means algorithm
num_clusters = 30

# List of files containing data
list_files = [
    'dict_STD_0_1.pkl', 'dict_STD_0_2.pkl', 'dict_STD_0_5.pkl',
    'dict_VACA_0_1.pkl', 'dict_VACA_0_2.pkl', 'dict_VACA_0_5.pkl'
]

# Corresponding names for the files
list_of_files_names = [
    'Standard 0.1', 'Standard 0.2', 'Standard 0.5',
    'Vaca 0.1', 'Vaca 0.2', 'Vaca 0.5'
]
#---------------------------------------------


# ------------------- LOADING DATA -------------------
# Initialize a dictionary to store data from all files
database = {}
size_of_groups = []

# Regular expression to extract numerical values from keys
regex = r"_([0-9]*\.[0-9]+)_"

# Visualization settings: color maps and markers for plotting
color_map = []
markers = ['o', 'o', 'o', '^', '^', '^', 'o', 's', '^', 'o', 's', '^', 'o', 's', '^']
map_type = ['Greens', 'Reds', 'Blues', 'Purples', 'Oranges', 'pink', 'Greens', 'Blues', 'Reds', 'Greens', 'Blues',
            'Reds']

# Load data from each file
for file_name in list_files:
    with open(file_name, 'rb') as input_file:
        # Track the size of the database before and after loading new data
        size_before = len(database)
        database.update(pickle.load(input_file))
        size_after = len(database)

        # Calculate the number of elements added
        elements_added = size_after - size_before
        size_of_groups.append(elements_added)

        # Extract numerical values from keys for color mapping
        input_file.seek(0)
        data = pickle.load(input_file)
        extracted_values = [float(key.split('_')[-1].rstrip('%')) for key in data.keys()]
        color_map.append(extracted_values)
        print(f"Data from {file_name} loaded successfully. Extracted values: {extracted_values}")

print(f"Sizes of groups loaded from files: {size_of_groups}")

# Extract names (keys) and fingerprints (values) from the database
def extract_number(key):
    parts = key.split('_')
    return int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 0
names = list(database.keys())
fingerprints = np.array(list(database.values()))
print(f"Total number of loaded structures: {len(names)}")
print(f"Fingerprint dimension: {len(fingerprints[0])}")

# ---------------- Transforming the initial data ----------------
# Standardize the fingerprints
scaler = StandardScaler()
fingerprints = scaler.fit_transform(fingerprints)

# Set weights for different parts of the fingerprints
# CrystalNN has 244 elements, followed by atomic masses, bond distances, and RDF
weight_crystalNN = 1
weight_element_mass_difference = 4
weight_average_bond_distance = 2
weight_RDF = 1

# Apply weights to the fingerprints
fingerprints[:, :244] *= weight_crystalNN
fingerprints[:, 244:248] *= weight_element_mass_difference
fingerprints[:, 248:252] *= weight_average_bond_distance
fingerprints[:, 252:] *= weight_RDF

#----------------------------------------------------------------



# ---------------- Saving mean and variance (for testing purposes) ----------------
# Calculate means and variances across all vectors
means = np.mean(fingerprints, axis=0)
variances = np.var(fingerprints, axis=0)

# Save means and variances to a file
with open('means_variance_vectors.txt', 'w') as f0:
    for idx, val in enumerate(means):
        f0.write(f"{means[idx]}\t{variances[idx]}\n")

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


all_indices = np.concatenate([part1_indices, part2_indices, part3_indices, part4_indices])
all_contributions = np.concatenate([contribution_vector_85[part1_indices],
                                     contribution_vector_85[part2_indices],
                                     contribution_vector_85[part3_indices],
                                     contribution_vector_85[part4_indices]])

# Create the color mapping for each segment
colors = ['b'] * len(part1_indices) + ['g'] * len(part2_indices) + ['r'] * len(part3_indices) + ['m'] * len(part4_indices)

# Create the bar graph
plt.figure(figsize=(16, 12))
plt.bar(all_indices, all_contributions, color=colors)


plt.xlabel('Feature Index', fontsize=25)
plt.ylabel('Contributions to 98% of Explained Variance', fontsize=25)
plt.title('Feature Contribution to the First 85 PCA Components', fontsize=25)


from matplotlib.patches import Patch
legend_handles = [
    Patch(color='b', label='CrystalNN\n(0-243)'),
    Patch(color='g', label='Atomic mass difference\n(244-247)'),
    Patch(color='r', label='Pair bond distances\n(248-251)'),
    Patch(color='m', label='RDF\n(252-351)')
]
plt.legend(handles=legend_handles, fontsize=16)

plt.grid(True)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

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
plt.show()


# For 3D t-SNE
tsne_3d = TSNE(n_components=3, random_state=0, perplexity=30)
tsne_3d_results = tsne_3d.fit_transform(principal_components)

fig_3d = plt.figure(figsize=(10, 8))
ax3 = fig_3d.add_subplot(111, projection='3d')
previous_val = 0
for idx, val in enumerate(size_of_groups):
    indecis_to_plot = val + previous_val
    indecis = range(previous_val, indecis_to_plot)
    previous_val = indecis_to_plot

    if idx == 0:
        sc = ax3.scatter(tsne_3d_results[indecis, 0], tsne_3d_results[indecis, 1], tsne_3d_results[indecis, 2],
                         label=list_of_files_names[idx], alpha=1, s=100, c=color_map[idx], cmap=map_type[idx], marker=markers[idx])
        cbar = plt.colorbar(sc, ax=ax3)
    else:
        sc = ax3.scatter(tsne_3d_results[indecis, 0], tsne_3d_results[indecis, 1], tsne_3d_results[indecis, 2],
                         label=list_of_files_names[idx], alpha=1, s=100, c=color_map[idx], cmap=map_type[idx], marker=markers[idx])

cbar.set_label('Color Scale of % lattice deformation', fontsize=15)
ax3.set_title('3D t-SNE visualization of PCA-transformed data', fontsize=15)
ax3.set_xlabel('t-SNE feature 1', fontsize=15)
ax3.set_ylabel('t-SNE feature 2', fontsize=15)
ax3.set_zlabel('t-SNE feature 3', fontsize=15)

# Highlight closest points to the centroids for each cluster
for cluster in range(num_clusters):
    indices = np.where(labels == cluster)[0]
    if len(indices) > 0:
        closest_point_index = indices[np.argmin([euclidean(fingerprints[idx], centroids[cluster]) for idx in indices])]
     #  ax3.scatter(tsne_3d_results[closest_point_index, 0], tsne_3d_results[closest_point_index, 1],
     #               tsne_3d_results[closest_point_index, 2], s=600, color='black', alpha=1)

#ax3.legend(fontsize=12)
plt.show()
