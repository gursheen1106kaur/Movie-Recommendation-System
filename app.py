import streamlit as st
import pickle
import pandas as pd
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered")

# Load the saved files
movies = pickle.load(open("models/movies.pkl", "rb"))

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if os.path.exists("models/similarity.pkl"):
    similarity = pickle.load(open("models/similarity.pkl", "rb"))
else:
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    similarity = cosine_similarity(tfidf_matrix)

st.title("🎬 Movie Recommendation System")
st.write("Welcome! 👋")
st.write("Select your favourite movie and get 5 similar movie recommendations.")

st.sidebar.header("About Project")

st.sidebar.write("""
This Movie Recommendation System suggests movies based on genre similarity.

Technologies Used:
- Python
- Pandas
- Scikit-learn
- Streamlit
""")

st.write("Welcome! Select a movie and I'll recommend similar movies.")
movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

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


if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write(movie)