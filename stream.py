import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------
# Streamlit Page Setup
# -------------------------------------
st.set_page_config(page_title="Amazon Music Clustering", layout="wide")
st.title("🎵 Amazon Music Clustering Dashboard")

# -------------------------------------
# Load Dataset
# -------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("single_genre_artists.csv")
    return df

df = load_data()
st.success(f"Dataset Loaded: {df.shape[0]} songs, {df.shape[1]} columns") # Display dataset shape

# -------------------------------------
# Feature Selection + Scaling
# -------------------------------------
features = [
    "danceability", "energy", "loudness", "speechiness", 
    "acousticness", "instrumentalness", "liveness", 
    "valence", "tempo", "duration_ms" 
]

X = df[features] # Feature matrix
scaler = StandardScaler() # Standardize features
X_scaled = scaler.fit_transform(X) # Scaled feature matrix

# -------------------------------------
# Sidebar Controls
# -------------------------------------
st.sidebar.header("⚙️ Clustering Settings") # Sidebar header
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=5, step=1) # Slider for k
run_clustering = st.sidebar.button("Run Clustering") # Button to run clustering

# -------------------------------------
# Run Clustering
# -------------------------------------
if run_clustering:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # KMeans model
    df["cluster"] = kmeans.fit_predict(X_scaled) # Assign clusters

    st.subheader("📊 Cluster Distribution") 
    cluster_counts = df["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # PCA for Visualization
    pca = PCA(n_components=2) # 2D PCA
    X_pca = pca.fit_transform(X_scaled) # PCA transformed data
    df["pca1"], df["pca2"] = X_pca[:,0], X_pca[:,1] # Add PCA components to df

    st.subheader("🎨 PCA Scatter Plot")
    fig, ax = plt.subplots(figsize=(8,6))
    for c in range(k):
        subset = df[df["cluster"] == c]
        ax.scatter(subset["pca1"], subset["pca2"], s=10, label=f"Cluster {c}")
    ax.set_title(f"PCA Visualization ({k} Clusters)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    st.pyplot(fig)

    # Cluster Profiles
    st.subheader("📌 Cluster Profiles (Mean Feature Values)")
    cluster_profile = df.groupby("cluster")[features].mean()
    st.dataframe(cluster_profile.style.highlight_max(axis=0))

    # Download clustered data
    st.subheader("⬇️ Download Clustered Dataset")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV with Clusters",
        data=csv,
        file_name="amazon_music_clusters.csv",
        mime="text/csv"
    )
