import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# -------------------------------
# Load Dataset
# -------------------------------
movies = pd.read_csv("dataset/movies.csv")

# Keep only required columns
movies = movies[["movieId", "title", "genres"]]

# Replace missing values
movies.fillna("", inplace=True)

# -------------------------------
# Create Similarity Matrix
# -------------------------------
with st.spinner("Preparing recommendation model..."):

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    similarity = cosine_similarity(tfidf_matrix)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    "Welcome! 👋 Select your favourite movie and get 5 similar movie recommendations."
)

# Sidebar
st.sidebar.header("About Project")

st.sidebar.write("""
### Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

This project recommends movies based on **genre similarity**
using **TF-IDF Vectorization** and **Cosine Similarity**.
""")

movie_list = movies["title"].values

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_list
)

if st.button("🎬 Recommend"):

    recommendations = recommend(selected_movie)

    st.success("Top 5 Recommended Movies")

    for i, movie in enumerate(recommendations, start=1):
        st.write(f"**{i}. {movie}**")