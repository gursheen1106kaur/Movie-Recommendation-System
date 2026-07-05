import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("dataset/movies.csv")

# Keep only required columns
movies = movies[['movieId', 'title', 'genres']]

# Replace empty values
movies.fillna('', inplace=True)

# Convert genres into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate similarity
similarity = cosine_similarity(tfidf_matrix)

# Save files
pickle.dump(movies, open("models/movies.pkl", "wb"))
pickle.dump(similarity, open("models/similarity.pkl", "wb"))

print("Model Created Successfully!")