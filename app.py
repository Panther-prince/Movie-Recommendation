import streamlit as st
import pandas as pd
import pickle
import joblib

# Load movies dataset
with open("movies.pkl", "rb") as f:
    movies_dict = pickle.load(f)
movies = pd.DataFrame(movies_dict)

# Load compressed similarity matrix
similarity = joblib.load("similarity_compressed.pkl")

# Recommendation function
def recommend(movie):
    if movie not in movies['title'].values:
        st.warning("⚠️ Movie not found in database!")
        return []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[movie_index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recommended_movies = [movies.iloc[i[0]].title for i in distances[1:6]]  # Top 5
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.success("Top 5 Recommended Movies:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
