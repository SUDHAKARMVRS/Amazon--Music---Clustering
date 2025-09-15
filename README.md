🎵 # Amazon Music Clustering Dashboard

📌 ##Project Overview

This project analyzes an Amazon Music dataset (single genre artists) and applies K-Means clustering to group songs based on audio features.
An interactive Streamlit dashboard is provided for clustering, visualization, and downloading results.

📂 ##Project Structure
├── Preprocess.ipynb         # Jupyter notebook for preprocessing steps
├── single_genre_artists.csv # Dataset used for clustering
├── stream.py                # Streamlit app (dashboard)
├── require.txt              # Required Python libraries

⚙️ ##Features

📊 ###Cluster songs using K-Means on audio features

🎨 ###2D PCA visualization of clusters

📌 ###Cluster profiles with average feature values

📈 ###Interactive controls to select number of clusters (k)

⬇️ ###Export clustered dataset as CSV

##🛠️ Installation

Clone repository / download project files

git clone <your-repo-link>
cd project-folder


##Create virtual environment (optional)

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


##Install dependencies

pip install -r require.txt

▶️ ##Run the Dashboard
streamlit run stream.py


After running, open 👉 http://localhost:8501 in your browser.

📊 ##Dataset Columns
Column	Description
danceability	Suitability for dancing (0–1)
energy	Intensity and activity (0–1)
loudness	Loudness in dB
speechiness	Spoken word presence (0–1)
acousticness	Acoustic probability (0–1)
instrumentalness	Instrumental probability (0–1)
liveness	Live performance probability (0–1)
valence	Positivity of mood (0–1)
tempo	Beats per minute
duration_ms	Song duration in ms
🚀 Example Use Cases

Build a music recommendation system

Explore genre trends and feature importance

Analyze how audio features influence clustering

📎 ##Requirements

See require.txt:

pandas
streamlit
matplotlib
seaborn
numpy

🧑‍💻## Author

Sudhakar.M


Project built for Data Science & Visualization practice using
🐍 Python | 🚀 Streamlit | 🤖 Scikit-learn | 📊 Pandas & Matplotlib
