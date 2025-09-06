import pandas as pd
import pickle
import requests
import streamlit as st

# Load the pre-trained model
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to fetch movie poster
def fetch_poster(movie_id):
    """
    Fetch the movie poster URL from TMDB API based on the movie ID.
    Returns None if API key is not set or poster not found.
    """
    api_key = st.secrets.get("TMDB_API_KEY", None)  # safer way
    if not api_key:
        return None

    try:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
        data = requests.get(url).json()
        return 'https://image.tmdb.org/t/p/w500/' + data.get('poster_path', '') if 'poster_path' in data else None
    except Exception:
        return None


# Function to recommend movies
def recommend(movie):
    """
    Recommend movies based on the input movie title.
    Returns movie names + poster URLs (poster may be None if not available).
    """
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters
