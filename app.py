import streamlit as st
import pandas as pd
import pickle
import joblib
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #e0e6ed;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Movie card styling */
    .movie-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(31, 38, 135, 0.5);
    }
    
    .movie-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 8px;
    }
    
    .movie-rank {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        float: left;
        margin-right: 15px;
        margin-top: 5px;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: black;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Success message styling */
    .success-banner {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 20px 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    
    .stat-item {
        text-align: center;
        color: white;
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching for better performance
@st.cache_data
def load_movies_data():
    try:
        with open("movies.pkl", "rb") as f:
            movies_dict = pickle.load(f)
        return pd.DataFrame(movies_dict)
    except FileNotFoundError:
        st.error("❌ Movies dataset not found! Please ensure 'movies.pkl' exists.")
        return pd.DataFrame()

@st.cache_data
def load_similarity_matrix():
    try:
        return joblib.load("similarity_compressed.pkl")
    except FileNotFoundError:
        st.error("❌ Similarity matrix not found! Please ensure 'similarity_compressed.pkl' exists.")
        return None

# Initialize session state
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# Load data
movies = load_movies_data()
similarity = load_similarity_matrix()

if movies.empty or similarity is None:
    st.stop()

# Recommendation function with enhanced features
def get_recommendations(movie_title, num_recommendations=5):
    """Get movie recommendations with enhanced error handling"""
    if movie_title not in movies['title'].values:
        return [], "Movie not found in database!"
    
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = list(enumerate(similarity[movie_index]))
        distances = sorted(distances, key=lambda x: x[1], reverse=True)
        
        recommended_movies = []
        for i in distances[1:num_recommendations+1]:
            movie_data = {
                'title': movies.iloc[i[0]].title,
                'similarity_score': round(i[1] * 100, 1)
            }
            recommended_movies.append(movie_data)
        
        return recommended_movies, None
    except Exception as e:
        return [], f"Error generating recommendations: {str(e)}"

# Header section
st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Discover your next favorite movie with AI-powered recommendations ✨</p>', unsafe_allow_html=True)

# Stats section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'''
    <div class="stat-item">
        <h3>🎭</h3>
        <h4>{len(movies)}</h4>
        <p>Total Movies</p>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="stat-item">
        <h3>🔥</h3>
        <h4>{len(st.session_state.recommendation_history)}</h4>
        <p>Recommendations Made</p>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="stat-item">
        <h3>⭐</h3>
        <h4>5</h4>
        <p>Top Picks</p>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="stat-item">
        <h3>🎯</h3>
        <h4>AI</h4>
        <p>Powered</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔍 Find Your Next Movie")
    
    # Movie selection with search
    search_term = st.text_input("🔎 Search for a movie:", placeholder="Type to search...")
    
    if search_term:
        filtered_movies = movies[movies['title'].str.contains(search_term, case=False, na=False)]['title'].tolist()
        if filtered_movies:
            selected_movie = st.selectbox("Select from search results:", filtered_movies, key="search_select")
        else:
            st.warning("No movies found matching your search.")
            selected_movie = st.selectbox("Or choose from all movies:", movies['title'].values, key="all_select")
    else:
        selected_movie = st.selectbox("Choose a movie you like:", movies['title'].values, key="main_select")
    
    # Number of recommendations slider
    num_recs = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5, key="num_recs")
    
    # Recommendation button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        recommend_btn = st.button("🎬 Get Recommendations", use_container_width=True)
        surprise_btn = st.button("🎲 Surprise Me!", use_container_width=True)

with col2:
    st.markdown("### 🎯 Quick Actions")
    
    # Random movie suggestion
    if st.button("🎰 Random Movie", use_container_width=True):
        random_movie = random.choice(movies['title'].tolist())
        st.info(f"🎬 How about: **{random_movie}**?")
    
    # Popular genres (mock data - you can enhance this with actual genre data)
    st.markdown("### 🎭 Trending Genres")
    genre_options = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi"]
    selected_genre = st.selectbox("Browse by genre:", genre_options)
    
    if st.button("Browse Genre", use_container_width=True):
        st.info(f"🔍 Browsing {selected_genre} movies...")

# Handle recommendations
if recommend_btn or surprise_btn:
    if surprise_btn:
        selected_movie = random.choice(movies['title'].tolist())
        st.info(f"🎲 Surprise pick: **{selected_movie}**")
    
    with st.spinner("🎬 Finding perfect matches..."):
        recommendations, error = get_recommendations(selected_movie, num_recs)
    
    if error:
        st.error(f"❌ {error}")
    elif recommendations:
        # Add to history
        st.session_state.recommendation_history.append({
            'movie': selected_movie,
            'timestamp': datetime.now().strftime("%H:%M"),
            'count': len(recommendations)
        })
        
        st.markdown('<div class="success-banner">🎉 Perfect matches found! Here are your recommendations:</div>', unsafe_allow_html=True)
        
        # Display recommendations in cards
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f'''
            <div class="movie-card">
                <div class="movie-rank">{i}</div>
                <div class="movie-title">{rec['title']}</div>
                <div style="color: #7f8c8d;">
                    <strong>Match Score:</strong> {rec['similarity_score']}% 
                    <span style="float: right;">⭐ Recommended</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("### 📊 Your Activity")
    
    if st.session_state.recommendation_history:
        st.markdown("#### Recent Searches")
        for item in st.session_state.recommendation_history[-5:]:
            st.markdown(f"🕐 {item['timestamp']} - **{item['movie']}** ({item['count']} recs)")
        
        if st.button("🗑️ Clear History"):
            st.session_state.recommendation_history = []
            st.rerun()
    else:
        st.info("No search history yet!")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Movie Recommendation System** uses advanced AI algorithms to analyze movie similarities and provide personalized recommendations based on your preferences.
    
    🎯 **Features:**
    - AI-powered recommendations
    - Search functionality
    - Surprise recommendations
    - Match scoring
    - History tracking
    """)
    
    st.markdown("---")
    st.markdown("### 🎪 Fun Stats")
    if movies is not None and not movies.empty:
        st.metric("Total Movies", len(movies))
        st.metric("Database Status", "✅ Active")
        st.metric("Recommendation Engine", "🚀 Online")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p>🎬 <strong>Movie Recommendation System</strong> - Your Personal Movie Discovery Assistant</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Powered by Machine Learning • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)