
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
import seaborn as sns

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

# Load and parse dataset
dataset = np.load("Dataset.npy")
spliter = lambda s: s.split(",")
dataset = np.array([spliter(x) for x in dataset])

# ========================================================================
# QUESTION 1: Find Unique Users and Items
# ========================================================================
print("\n" + "="*40 + "\nQUESTION 1: Unique Users/Items\n" + "="*40)
if os.path.exists(os.path.join(datafolder, "dataframe.pkl")):
    dataframe = pd.read_pickle(os.path.join(datafolder, "dataframe.pkl"))
else:
    dataframe = pd.DataFrame(dataset, columns=["user", "item", "rating", "date"])
    # Clean data
    dataframe["user"] = dataframe["user"].str.replace("ur", "").astype(np.int64)
    dataframe["item"] = dataframe["item"].str.replace("tt", "").astype(np.int64)
    dataframe["rating"] = dataframe["rating"].astype(np.int64)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
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
    ratings_num_df = dataframe.groupby("user")["rating"].count().reset_index(name="ratings_num")
    ratings_num_df.to_pickle(os.path.join(datafolder, "ratings_num_df.pkl"))

if os.path.exists(os.path.join(datafolder, "ratings_span_df.pkl")):
    ratings_span_df = pd.read_pickle(os.path.join(datafolder, "ratings_span_df.pkl"))
else:
    ratings_span_df = dataframe.groupby("user")["date"].agg(lambda x: (x.max() - x.min()).days).reset_index(name="ratings_span")
    ratings_span_df.to_pickle(os.path.join(datafolder, "ratings_span_df.pkl"))

# Merge and filter
ratings_df = ratings_num_df.merge(ratings_span_df, on="user")
R_min, R_max = 100, 300
filtered_users_df = ratings_df.query(f"{R_min} <= ratings_num <= {R_max}")

# Create final filtered dataset
final_df = dataframe[dataframe["user"].isin(filtered_users_df["user"])].copy()
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
# QUESTION 4: Preference Vectors R
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

# Initialize a sparse matrix R (n x m) where n = |Û|, m = |Î|
R = lil_matrix((len(user_ids), len(item_ids)), dtype=np.float32)

# Fill the matrix with ratings (only positive values, > 0)
for _, row in final_df.iterrows():
    if row['rating'] > 0:  # Ensures that R_j(k) = 0 if there’s no rating
        user_idx = user_to_idx[row['user']]
        item_idx = item_to_idx[row['item']]
        R[user_idx, item_idx] = row['rating']

# Convert to CSR format for efficiency
R_dense = R.tocsr()

# Save the preference matrix
np.save(os.path.join(datafolder, "preference_matrix.npy"), R_dense)

# Additional output for inspection
print(f"Preference matrix shape: {R_dense.shape}")
print(f"Non-zero elements: {R_dense.count_nonzero()}")
print(f"Density: {(R_dense.count_nonzero() / (R_dense.shape[0] * R_dense.shape[1])) * 100:.2f}%")

# Print the matrix in a readable format
user_vector = R_dense[0].toarray().flatten()
non_zero_indices = np.where(user_vector != 0)[0]
print(f"Rated movie indices: {non_zero_indices}")
print(f"Ratings: {user_vector[non_zero_indices]}")


# ========================================================================
# Clustering question a 
# ========================================================================
print("\n" + "="*40 + "\n Cluster question: a \n" + "="*40)

