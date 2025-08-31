import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
import seaborn as sns

from sklearn.metrics import pairwise_distances, silhouette_score,accuracy_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import TruncatedSVD
import time

# ========================================================================
# Function Definitions
# ========================================================================
def plot_histogram(series, title_str):
    """Plot and save histograms for data distribution analysis"""
    n, bins, patches = plt.hist(series, bins='auto', color='red', alpha=0.7, rwidth=0.75)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(series.name)
    plt.ylabel('Frequency')
    plt.title(title_str)
    plt.ylim(ymax=np.ceil(n.max() / 10) * 10 if n.max() % 10 else n.max() + 10)
    plt.savefig(os.path.join(FIGURES_PATH, f"{series.name}.png"), dpi=100, bbox_inches='tight')
    plt.show()


# ========================================================================
# Initialization and Data Loading
# ========================================================================
# Configure paths
FIGURES_PATH = "figures"
datafolder = "datafiles"
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(datafolder, exist_ok=True)

MODE = "spec"  # change to "mirror" - spec for keeping ratings >0 

# Load and parse dataset 
raw = np.load("Dataset.npy", allow_pickle=True)
decoded = []
for x in raw:
    if isinstance(x, bytes):
        decoded.append(x.decode("utf-8"))
    else:
        decoded.append(str(x))
spliter = lambda s: s.split(",")
dataset = np.array([spliter(s) for s in decoded])
# print(dataset)

# ========================================================================
# QUESTION 1: Find Unique Users and Items
# ========================================================================
print("\n" + "="*40 + "\nQUESTION 1: Unique Users/Items\n" + "="*40)
if os.path.exists(os.path.join(datafolder, "dataframe.pkl")):
    dataframe = pd.read_pickle(os.path.join(datafolder, "dataframe.pkl"))
    # Ensure df_for_stats is defined the same way as when building from raw
    # (MODE controls whether we dedupe and keep only positive ratings)
    if MODE == "spec":
        df_for_stats = dataframe.sort_values("date").drop_duplicates(subset=["user", "item"], keep="last")
        df_for_stats = df_for_stats[df_for_stats["rating"] > 0].copy()
    else:
        df_for_stats = dataframe.copy()
else:
    dataframe = pd.DataFrame(dataset, columns=["user", "item", "rating", "date"])

    # Clean data 
    dataframe["user"] = dataframe["user"].str.replace("ur", "", regex=False).str.strip()
    dataframe["item"] = dataframe["item"].str.replace("tt", "", regex=False).str.strip()
    dataframe["user"] = pd.to_numeric(dataframe["user"], errors="coerce").astype("Int64")
    dataframe["item"] = pd.to_numeric(dataframe["item"], errors="coerce").astype("Int64")
    dataframe["rating"] = pd.to_numeric(dataframe["rating"], errors="coerce")
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")

    # Drop rows with missing essential fields
    dataframe = dataframe.dropna(subset=["user", "item", "rating", "date"])

    # Prepare two possible dataframes for stats depending on MODE
    if MODE == "spec":  # MODE='mirror'
        # strict: drop duplicates per (user,item) keeping last, then keep only positive ratings
        df_for_stats = dataframe.sort_values("date").drop_duplicates(subset=["user", "item"], keep="last")
        df_for_stats = df_for_stats[df_for_stats["rating"] > 0].copy()
    else:
        # mirror: use cleaned data as-is 
        df_for_stats = dataframe.copy()

    # Save cleaned dataframe
    dataframe.to_pickle(os.path.join(datafolder, "dataframe.pkl"))
    
# Report initial statistics
unique_users = dataframe["user"].unique()
unique_items = dataframe["item"].unique()
print(f"Initial unique users (U): {len(unique_users)}")
print(f"Initial unique items (I): {len(unique_items)}")
print(f"Total ratings: {dataframe.shape[0]}")

# ========================================================================
# QUESTION 2: Filter Users and Items
# ========================================================================
print("\n" + "="*40 + "\nQUESTION 2: Filter Users/Items\n" + "="*40)
# Calculate ratings statistics
if os.path.exists(os.path.join(datafolder, "ratings_num_df.pkl")):
    ratings_num_df = pd.read_pickle(os.path.join(datafolder, "ratings_num_df.pkl"))
else:
    # Use df_for_stats according to MODE:
    # - spec: df_for_stats is deduped+positive -> nunique gives |φ(u)|
    # - mirror: df_for_stats is cleaned raw -> count() 
    if MODE == "spec":
        ratings_num_df = df_for_stats.groupby("user")["item"].nunique().reset_index(name="ratings_num")
    else:
        ratings_num_df = df_for_stats.groupby("user")["rating"].count().sort_values(ascending=False).reset_index(name="ratings_num")
    ratings_num_df.to_pickle(os.path.join(datafolder, "ratings_num_df.pkl"))
 
if os.path.exists(os.path.join(datafolder, "ratings_span_df.pkl")):
    ratings_span_df = pd.read_pickle(os.path.join(datafolder, "ratings_span_df.pkl"))
else:
    # Compute span on df_for_stats (consistent with MODE)
    ratings_span_df = df_for_stats.groupby("user")["date"].agg(lambda x: x.max() - x.min()).reset_index(name="ratings_span")
    ratings_span_df.to_pickle(os.path.join(datafolder, "ratings_span_df.pkl"))

# Merge and filter
ratings_df = ratings_num_df.merge(ratings_span_df, on="user")
# Convert ratings_span to numeric days if needed
if pd.api.types.is_timedelta64_dtype(ratings_df["ratings_span"]):
    ratings_df["ratings_span"] = ratings_df["ratings_span"].dt.days
R_min, R_max = 100, 300
filtered_users_df = ratings_df.query(f"{R_min} <= ratings_num <= {R_max}")
# print(ratings_num_df['ratings_num'].value_counts().head())

# Create final filtered dataset
final_df = df_for_stats[df_for_stats["user"].isin(filtered_users_df["user"])].copy()
final_items = final_df["item"].unique()

print(f"Filtered users (Û): {len(filtered_users_df)}")
print(f"Filtered items (Î): {len(final_items)}")
print(f"Remaining ratings: {len(final_df)}")

# ========================================================================
# QUESTION 3: Create Histograms
# ========================================================================
print("\n" + "="*40 + "\nQUESTION 3: Histograms\n" + "="*40)
plot_histogram(filtered_users_df["ratings_num"], "Ratings per User Distribution")
plot_histogram(filtered_users_df["ratings_span"], "Rating Time Span Distribution")

# ========================================================================
# QUESTION 4: Preference Matrix
# ========================================================================
print("\n" + "="*40 + "\nQUESTION 4: Preference Matrix\n" + "="*40)

# Keep only the latest rating per (user, item)
final_df = final_df.sort_values("date").drop_duplicates(subset=["user", "item"], keep="last")

# Create unique identifiers for users and movies
final_users = final_df["user"].unique()
final_items = final_df["item"].unique()

# Create mappings with sorted lists
user_ids = np.sort(final_users)
item_ids = np.sort(final_items)
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {i: j for j, i in enumerate(item_ids)}

R = lil_matrix((len(user_ids), len(item_ids)), dtype=np.float32)

# Fill the matrix with ratings (only positive values, > 0)
for _, row in final_df.iterrows():
    if row['rating'] > 0:  # Ensures that R_j(k) = 0 if there’s no rating
        user_idx = user_to_idx[row['user']]
        item_idx = item_to_idx[row['item']]
        R[user_idx, item_idx] = row['rating']

# Convert to CSR format for efficiency
R_dense = R.tocsr()

# Save the preference matrix (use sparse format)
from scipy.sparse import save_npz
save_npz(os.path.join(datafolder, "preference_matrix.npz"), R_dense)

# Additional output for inspection
print(f"Preference matrix shape: {R_dense.shape}")
print(f"Non-zero elements: {R_dense.count_nonzero()}")
print(f"Density: {(R_dense.count_nonzero() / (R_dense.shape[0] * R_dense.shape[1])) * 100:.2f}%")

# Print the matrix in a readable format
idx = 500  
# Guard against index out of range
if idx < 0 or idx >= R_dense.shape[0]:
    print(f"Index {idx} is out of range (0..{R_dense.shape[0]-1}). Skipping example print.")
else:
    user_vector = R_dense[idx].toarray().flatten()
    non_zero_indices = np.where(user_vector != 0)[0]
    print(f"Rated movie indices: {non_zero_indices}")
    print(f"Movie IDs: {item_ids[non_zero_indices]}")
    print(f"Ratings: {user_vector[non_zero_indices]}")

######## Testing ########
# user_id = user_ids[500]  
# item_id = 394894
# if item_id in item_ids:
#     item_idx = item_to_idx[item_id]
#     rating_matrix = R_dense[user_idx, item_idx]
#     df_rating = final_df[(final_df['user'] == user_id) & (final_df['item'] == item_id)]
#     if not df_rating.empty:
#         rating_df = df_rating['rating'].values[0]
#     else:
#         rating_df = 0
#     print(f"User ID: {user_id}, Item ID: {item_id}")
#     print(f"Rating from matrix: {rating_matrix}")
#     print(f"Rating from DataFrame: {rating_df}")
# else:
#     print("item_id is not existing.")

# # ========================================================================
# # QUESTION Clustering:
# # ========================================================================
# print("\n" + "="*40 + "\nQUESTION Clustering:\n" + "="*40)

def custom_euclidean(Ru, Rv):
    Ru = np.asarray(Ru, dtype=np.float64)
    Rv = np.asarray(Rv, dtype=np.float64)
    lambda_u = (Ru > 0).astype(np.int8)
    lambda_v = (Rv > 0).astype(np.int8)
    common_mask = lambda_u & lambda_v
    if common_mask.sum() == 0:
        return float(1e6)
    diff = Ru - Rv
    distance = np.sqrt(np.sum((diff * common_mask) ** 2))
    return float(distance)

def custom_cosine(Ru, Rv):
    Ru = np.asarray(Ru, dtype=np.float64)
    Rv = np.asarray(Rv, dtype=np.float64)
    lambda_u = (Ru > 0).astype(np.int8)
    lambda_v = (Rv > 0).astype(np.int8)
    common_mask = lambda_u & lambda_v
    if common_mask.sum() == 0:
        return 1.0
    Ru_c = Ru * common_mask
    Rv_c = Rv * common_mask
    num = float(np.dot(Ru_c, Rv_c))
    denom_u = float(np.dot(Ru_c, Ru_c))
    denom_v = float(np.dot(Rv_c, Rv_c))
    if denom_u <= 0 or denom_v <= 0:
        return 1.0
    cosine_sim = num / (np.sqrt(denom_u) * np.sqrt(denom_v))
    return float(1.0 - abs(cosine_sim))

# compute full pairwise distance matrix using CSR rows (only on common items)
def compute_distance_matrix(R_csr, metric="euclidean"):
    """Compute full symmetric distance matrix for users (rows of R_csr).
    metric: "euclidean" or "cosine". Distances computed only on common rated items (R>0).
    WARNING: O(n^2) time & memory."""
    n = R_csr.shape[0]
    # Pre-extract indices/data per row
    rows_idx = [R_csr[i].indices for i in range(n)]
    rows_data = [R_csr[i].data for i in range(n)]
    dist = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        idx_i = rows_idx[i]
        dict_i = dict(zip(idx_i, rows_data[i]))
        for j in range(i+1, n):
            idx_j = rows_idx[j]
            # intersection of indices
            common = np.intersect1d(idx_i, idx_j, assume_unique=True)
            if common.size == 0:
                if metric.startswith("euc"):
                    d = 1e6
                else:
                    d = 1.0
            else:
                vi = np.array([dict_i[k] for k in common], dtype=np.float64)
                dict_j = dict(zip(idx_j, rows_data[j]))
                vj = np.array([dict_j[k] for k in common], dtype=np.float64)
                if metric.startswith("euc"):
                    d = custom_euclidean(vi, vj)
                else:
                    d = custom_cosine(vi, vj)
            dist[i, j] = dist[j, i] = d
        # optional progress print
        if (i+1) % 100 == 0 or i == n-1:
            print(f"Distances: processed row {i+1}/{n}")
    t1 = time.time()
    print(f"Computed distance matrix ({metric}) for n={n} in {t1-t0:.1f}s")
    return dist

# reduce to 2D for plotting (TruncatedSVD on sparse -> PCA to 2D)
def embed_2d(R_csr, svd_components=50):
    n_items = R_csr.shape[1]
    if n_items <= 2:
        return R_csr.toarray()
    svd = TruncatedSVD(n_components=min(svd_components, max(2, n_items-1)), random_state=42)
    X_svd = svd.fit_transform(R_csr)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_svd)
    return X2

# Run clustering experiments (Euclidean & Cosine) for k=1..10
n_users = R_dense.shape[0]
print(f"Clustering step: users={n_users}")

# Compute both distance matrices (full)
dist_eu = compute_distance_matrix(R_dense, metric="euclidean")
dist_cos = compute_distance_matrix(R_dense, metric="cosine")

# Post-process distance matrices: enforce symmetry, non-negativity and zero diagonal
for D in (dist_eu, dist_cos):
    # average with transpose to remove tiny asymmetry
    D[:] = 0.5 * (D + D.T)
    # set diagonal exactly zero
    np.fill_diagonal(D, 0.0)
    # clip tiny negative rounding errors
    np.clip(D, 0.0, None, out=D)
    
# 2D embedding for visualization
X2 = embed_2d(R_dense, svd_components=50)

# For each metric run KMedoids for k=1..10, collect inertia and silhouette
def run_kmedoids_experiments(dist_matrix, metric_name="euclidean", k_max=10):
    ks = list(range(1, k_max+1))
    inertias = []
    silhouettes = []
    labels_dict = {}
    for k in ks:
        model = KMedoids(n_clusters=k, metric='precomputed', random_state=1)
        model.fit(dist_matrix)
        inertias.append(model.inertia_)
        lbls = model.labels_
        labels_dict[k] = lbls
        if k > 1:
            try:
                sil = silhouette_score(dist_matrix, lbls, metric='precomputed')
            except Exception:
                sil = np.nan
        else:
            sil = np.nan
        silhouettes.append(sil)
        print(f"{metric_name} k={k}: inertia={model.inertia_:.3f}, silhouette={sil:.4f}")
    # Elbow plot (save)
    plt.figure(figsize=(6,4))
    plt.plot(ks, inertias, 'bx-')
    plt.xlabel('k'); plt.ylabel('Distortion (inertia)')
    plt.title(f'Elbow ({metric_name})')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_PATH, f"elbow_{metric_name}.png"), dpi=100, bbox_inches='tight')
    plt.show()
    # Silhouette plot (save)
    plt.figure(figsize=(6,4))
    plt.plot(ks, silhouettes, 'rx-')
    plt.xlabel('k'); plt.ylabel('Silhouette score')
    plt.title(f'Silhouette ({metric_name})')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_PATH, f"silhouette_{metric_name}.png"), dpi=100, bbox_inches='tight')
    plt.show()
    return labels_dict

labels_eu = run_kmedoids_experiments(dist_eu, metric_name="euclidean", k_max=10)
labels_cos = run_kmedoids_experiments(dist_cos, metric_name="cosine", k_max=10)

# Visualize clusters for selected k values (2,3,4) for both metrics
for metric_name, labels_dict in [("euclidean", labels_eu), ("cosine", labels_cos)]:
    for k in (2,3,4):
        if k not in labels_dict: 
            continue
        lbls = labels_dict[k]
        plt.figure(figsize=(8,6))
        plt.scatter(X2[:,0], X2[:,1], c=lbls, cmap='tab10', s=12, alpha=0.8)
        plt.title(f'KMedoids {metric_name} k={k}')
        plt.xlabel('Dim1'); plt.ylabel('Dim2'); plt.grid(True)
        fname = os.path.join(FIGURES_PATH, f"clusters_{metric_name}_k{k}.png")
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        plt.show()
        # save labels
        np.save(os.path.join(datafolder, f"clusters_{metric_name}_k{k}.npy"), lbls)
print("Clustering completed.")


