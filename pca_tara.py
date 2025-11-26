import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


# Open and read the binary file containing int32 vectors of dimension 2048
with open('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/tara/tara_vectors.bin', 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.int32)

# Reshape the data into vectors of dimension 2048
num_vectors = len(data) // 2048
vectors = data.reshape(num_vectors, 2048)

# Divide all vectors by sqrt of 2048
vectors = vectors / np.sqrt(2048)

# Load metadata
metadata = pd.read_csv('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/tara/SraRunTable.csv')
print(f"Metadata shape: {metadata.shape}")
print(f"Number of samples in metadata: {len(metadata)}")
print(f"Number of vectors: {num_vectors}")
# Convert Depth column to numeric, handling any non-numeric values
metadata['Depth'] = pd.to_numeric(metadata['Depth'], errors='coerce')
metadata['Latitude_End'] = pd.to_numeric(metadata['Latitude_End'], errors='coerce')
# Convert Latitude_End to its absolute value
metadata['Latitude_End'] = metadata['Latitude_End'].abs()
metadata['Nitrate_Sensor'] = pd.to_numeric(metadata['Nitrate_Sensor'], errors='coerce')
metadata['Oxygen_Sensor'] = pd.to_numeric(metadata['Oxygen_Sensor'], errors='coerce')
metadata['Salinity_Sensor'] = pd.to_numeric(metadata['Salinity_Sensor'], errors='coerce')
print("qlksd ", metadata['Salinity_Sensor'])

# Load vector names
with open('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/tara/vector_norms.txt', 'r') as f:
    vector_names = []
    for line in f:
        fields = line.strip().split()
        if fields:  # Check if line is not empty
            vector_names.append(fields[0])

def perform_pca_analysis(vectors, metadata, vector_names, metadata_column='Depth', filter_range=(-1, 10000)):
    """
    Perform PCA analysis and plot results colored by specified metadata column.
    
    Parameters:
    - vectors: numpy array of vectors
    - metadata: pandas DataFrame with metadata
    - vector_names: list of vector names
    - metadata_column: string, column name in metadata to use for coloring
    - filter_range: tuple, (min, max) values to filter the metadata column
    """
    print(f"Number of vector names: {len(vector_names)}")
    
    # Match vector names to metadata by sample names
    # Extract sample names from vector names (assuming they contain the Run column values)
    vector_sample_names = [name.split('_')[0] if '_' in name else name for name in vector_names]

    # Create a mapping from Run to metadata column
    run_to_metadata = dict(zip(metadata['Run'], metadata[metadata_column]))

    # Get metadata values for each vector
    metadata_values = []
    for sample_name in vector_sample_names:
        value = run_to_metadata.get(sample_name, None)
        metadata_values.append(value)

    # Filter out vectors that do not have valid metadata values
    valid_indices = [i for i, value in enumerate(metadata_values) 
                    if value is not None and filter_range[0] < value < filter_range[1]]
    filtered_vectors = vectors[valid_indices]
    filtered_metadata_values = [metadata_values[i] for i in valid_indices]
    filtered_vector_names = [vector_names[i] for i in valid_indices]

    print(f"After filtering, number of vectors: {len(filtered_vectors)}")
    print(f"After filtering, number of {metadata_column} values: {len(filtered_metadata_values)}")

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(filtered_vectors)

    # Show PCA results
    print(f"Original shape: {filtered_vectors.shape}")
    print(f"PCA result shape: {pca_result.shape}")
    print(f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative explained variance (first 10 components): {np.cumsum(pca.explained_variance_ratio_[:10])}")

    # Plot the first two PCA components colored by metadata
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=filtered_metadata_values, alpha=0.6, cmap='viridis')

    # Add colorbar
    plt.colorbar(scatter, label=metadata_column)

    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title(f'PCA: First Two Principal Components (colored by {metadata_column})')
    plt.grid(True, alpha=0.3)
    plt.show()

# Call the function with Depth as the metadata column
perform_pca_analysis(vectors, metadata, vector_names, metadata_column='Salinity_Sensor', filter_range=(-10000, 10000))