import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


# Open and read the binary file containing int32 vectors of dimension 2048
with open('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/tara/combined_vectors.bin', 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.int32)

# Reshape the data into vectors of dimension 2048
num_vectors = len(data) // 2048
vectors = data.reshape(num_vectors, 2048)

# Divide all vectors by sqrt of 2048
vectors = vectors / np.sqrt(2048)

# Load vector names
with open('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/tara/vector_norms_combined.txt', 'r') as f:
    vector_names = []
    for line in f:
        fields = line.strip().split()
        if fields:  # Check if line is not empty
            vector_names.append(fields[0])

def perform_pca_analysis(vectors, vector_names):
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

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(vectors)

    # Show PCA results
    print(f"PCA result shape: {pca_result.shape}")
    print(f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative explained variance (first 10 components): {np.cumsum(pca.explained_variance_ratio_[:10])}")

    # Plot the first two PCA components colored by vector type
    plt.figure(figsize=(12, 8))
    
    # Split vectors into Tara (first 852) and others
    tara_indices = range(852)
    other_indices = range(852, len(pca_result))
    
    # Plot Tara vectors
    plt.scatter(pca_result[tara_indices, 0], pca_result[tara_indices, 1], 
                alpha=0.6, c='red', label='Ocean metagenome')
    
    # Plot other vectors
    plt.scatter(pca_result[other_indices, 0], pca_result[other_indices, 1], 
                alpha=0.6, c='blue', label='Gut metagenome')

    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.3f})', fontsize=18)
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.3f})', fontsize=18)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.show()

# Call the function with Depth as the metadata column
perform_pca_analysis(vectors, vector_names)