import os
import sys
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_vectors(folder):
    # Read dimension
    dim_path = os.path.join(folder, "dimension.txt")
    with open(dim_path, "r") as f:
        dim = int(f.read().strip())

    # Read vectors.bin
    vectors_path = os.path.join(folder, "vectors.bin")
    vectors = np.fromfile(vectors_path, dtype=np.int32)
    if vectors.size % dim != 0:
        raise ValueError("vectors.bin size is not a multiple of dimension")
    vectors = vectors.reshape(-1, dim)

    # Read vector norms
    norms_path = os.path.join(folder, "vector_norms.txt")
    norms = []
    with open(norms_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            norm = float(parts[1])
            norms.append(norm)
    norms = np.array(norms)

    # Filter vectors with norm >= 10
    mask = norms >= 10
    vectors = vectors[mask]

    # Load names and filter using the same mask
    names = []
    with open(os.path.join(folder, "vector_norms.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            names.append(parts[0])
    names = np.array(names)
    names = names[mask]

    return vectors, names

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder>")
        sys.exit(1)
    folder = sys.argv[1]
    vectors, names = load_vectors(folder)
    print("vectors loaded, I have ", len(vectors), " vectors")

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(vectors)
    print("pca computed")

    # Load big_vectors.bin and project onto PCA space
    big_vectors_path = os.path.join(folder, "big_vectors.bin")
    if os.path.exists(big_vectors_path):
        # Only load the first 500000 vectors
        dim = vectors.shape[1]
        max_vectors = 500000
        count = min(max_vectors, os.path.getsize(big_vectors_path) // (4 * dim))
        big_vectors = np.fromfile(big_vectors_path, dtype=np.int32, count=count * dim)
        if big_vectors.size % dim != 0:
            raise ValueError("big_vectors.bin size is not a multiple of dimension")
        big_vectors = big_vectors.reshape(-1, dim)
        big_vectors_pca = pca.transform(big_vectors)
        # Plot big_vectors in the PCA space
        plt.scatter(big_vectors_pca[:, 0], big_vectors_pca[:, 1], alpha=0.3, color='red', label='big_vectors')
        plt.legend()
    else:
        print("big_vectors.bin not found, skipping projection.")

    print("all vectors projected")

    # Plot the first two PCA axes
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 1], pca_result[:, 2], alpha=0.7)
    for i, name in enumerate(names):
        plt.annotate(name, (pca_result[i, 1], pca_result[i, 2]), fontsize=8, alpha=0.7)
    plt.xlabel(f"PCA Axis 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
    plt.ylabel(f"PCA Axis 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
    plt.title("PCA: First Two Axes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()