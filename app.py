import streamlit as st
import pandas as pd
import pickle
import joblib
import random
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="NETFLIX - Movie Recommendations",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Netflix-style CSS that matches the HTML version
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: #000000;
        font-family: 'Netflix Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        color: white;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Netflix Header */
    .netflix-header {
        background: rgba(0, 0, 0, 0.9);
        padding: 15px 0;
        margin-bottom: 0;
        border-bottom: 1px solid #333;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .netflix-logo {
        color: #e50914;
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .netflix-nav {
        text-align: center;
        color: #e5e5e5;
        font-size: 14px;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(
            rgba(0, 0, 0, 0.4), 
            rgba(0, 0, 0, 0.6)
        ),
        radial-gradient(circle at 30% 40%, rgba(229, 9, 20, 0.1) 0%, transparent 70%);
        padding: 120px 60px 80px;
        text-align: center;
        margin-bottom: 40px;
        border-radius: 0;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 24px;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: white;
        margin-bottom: 40px;
        line-height: 1.4;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin: 40px 0 30px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e50914;
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(45deg, #e50914, #ff6b6b);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        color: white;
    }
    
    .section-title {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    
    /* Custom Streamlit Widget Styling */
    .stSelectbox > div > div {
        background: #333 !important;
        border: 2px solid #444 !important;
        color: white !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #e50914 !important;
        box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.1) !important;
    }
    
    .stTextInput > div > div > input {
        background: #333 !important;
        border: 2px solid #444 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 16px 18px !important;
        font-size: 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #e50914 !important;
        box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #777 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #e50914, #ff1e2d) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 14px 28px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #f40612, #ff2e3d) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(229, 9, 20, 0.5) !important;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div > div {
        background: #e50914 !important;
    }
    
    .stSlider > div > div > div {
        background: #333 !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    .stSlider {
        padding: 20px 0 !important;
    }
    
    /* Stats Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 25px;
        margin: 40px 0;
        padding: 20px 0;
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 40px 25px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e50914, #ff6b6b);
    }
    
    .stat-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #e50914;
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.2);
    }
    
    .stat-icon {
        font-size: 48px;
        margin-bottom: 20px;
        display: block;
    }
    
    .stat-number {
        font-size: 42px;
        font-weight: 800;
        color: #e50914;
        margin-bottom: 12px;
        display: block;
    }
    
    .stat-label {
        color: #b3b3b3;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Movie Cards */
    .movie-card {
        background: linear-gradient(145deg, #1a1a1a, #0f0f0f);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e50914, #ff6b6b);
    }
    
    .movie-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: #e50914;
        box-shadow: 0 15px 40px rgba(229, 9, 20, 0.3);
    }
    
    .movie-rank {
        position: absolute;
        top: 20px;
        right: 20px;
        background: linear-gradient(45deg, #e50914, #ff1e2d);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 16px;
        box-shadow: 0 4px 10px rgba(229, 9, 20, 0.4);
    }
    
    .movie-title {
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
        padding-right: 60px;
        line-height: 1.3;
    }
    
    .movie-match {
        color: #46d369;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .movie-label {
        color: #999;
        font-size: 14px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #111111 0%, #0a0a0a 100%);
    }
    
    .sidebar-title {
        color: #e50914;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 2px solid #e50914;
        padding-bottom: 15px;
    }
    
    .history-item {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(20, 20, 20, 0.8) 100%);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        border-left: 4px solid #e50914;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(30, 30, 30, 0.9) 100%);
        transform: translateX(5px);
    }
    
    .history-movie {
        color: white;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 6px;
    }
    
    .history-details {
        color: #999;
        font-size: 12px;
    }
    
    /* Success/Error Messages */
    .success-message {
        background: linear-gradient(90deg, #46d369, #4ecdc4);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 20px 0;
        border: none;
    }
    
    .error-message {
        background: linear-gradient(90deg, #ef5350, #e53935);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 20px 0;
        border: none;
    }
    
    .info-message {
        background: linear-gradient(90deg, #42a5f5, #1e88e5);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 20px 0;
        border: none;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #111 0%, #000 100%);
        border-top: 1px solid #333;
        padding: 50px 20px;
        text-align: center;
        margin-top: 60px;
    }
    
    .footer-logo {
        color: #e50914;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 15px;
    }
    
    .footer-text {
        color: #999;
        font-size: 16px;
        margin-bottom: 8px;
    }
    
    .footer-subtext {
        color: #666;
        font-size: 14px;
    }
    
    /* Form Labels */
    .stTextInput label, .stSelectbox label, .stSlider label {
        color: #b3b3b3 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin-bottom: 8px !important;
    }
    
    /* Loading Spinner */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #333;
        border-top: 5px solid #e50914;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 36px;
        }
        
        .stats-container {
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .stat-card {
            padding: 30px 15px;
        }
        
        .stat-icon {
            font-size: 36px;
        }
        
        .stat-number {
            font-size: 28px;
        }
        
        .hero-section {
            padding: 80px 30px 60px;
        }
        
        .section-title {
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
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
if 'total_recommendations' not in st.session_state:
    st.session_state.total_recommendations = 0
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = []
if 'recommended_for' not in st.session_state:
    st.session_state.recommended_for = ""

# Load data
movies = load_movies_data()
similarity = load_similarity_matrix()

if movies.empty or similarity is None:
    st.stop()

# Recommendation function
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

# Netflix Header
st.markdown("""
<div class="netflix-header">
    <div class="netflix-logo">NETFLIX</div>
    <div class="netflix-nav">Movie Recommendations System</div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Unlimited movies, TV shows, and more.</h1>
    <p class="hero-subtitle">Discover your next favorite movie with AI-powered recommendations tailored just for you.</p>
</div>
""", unsafe_allow_html=True)


# Stats Section
st.markdown("""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-icon">🎭</div>
        <div class="stat-number">{:,}</div>
        <div class="stat-label">MOVIES AVAILABLE</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🔥</div>
        <div class="stat-number">{}</div>
        <div class="stat-label">RECOMMENDATIONS MADE</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⭐</div>
        <div class="stat-number">98%</div>
        <div class="stat-label">SATISFACTION RATE</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-number">AI</div>
        <div class="stat-label">POWERED ENGINE</div>
    </div>
</div>
""".format(len(movies), st.session_state.total_recommendations), unsafe_allow_html=True)


# Sidebar Content
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 My List & History</div>', unsafe_allow_html=True)
    
    # Display recommendation history
    if st.session_state.recommendation_history:
        st.markdown("### 📚 Recent Searches")
        for item in st.session_state.recommendation_history[-10:]:
            if st.button(f"{item['movie']}", key=f"history_{item['movie']}_{item['timestamp']}", help=f"🕐 {item['timestamp']} • {item['count']} recs"):
                # Select this movie when clicked
                st.session_state.selected_from_history = item['movie']
                st.rerun()
        
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.recommendation_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No search history yet!\n\nStart by getting some recommendations.")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Movies", f"{len(movies):,}")
    with col2:
        st.metric("Searches", len(st.session_state.recommendation_history))

# Main Content Area
# Search Section
st.markdown("""
<div class="section-header">
    <div class="section-icon">🔍</div>
    <h2 class="section-title">Find Your Next Movie</h2>
</div>
""", unsafe_allow_html=True)

# Search Form
col1, col2 = st.columns([2, 1])

with col1:
    search_term = st.text_input(
        "Search Movies", 
        placeholder="Type to search movies...",
        key="movie_search_input",
        value=st.session_state.get('selected_from_history', '')
    )

with col2:
    # Filter movies based on search or show all
    if search_term:
        filtered_movies = movies[movies['title'].str.contains(search_term, case=False, na=False)]['title'].tolist()
        if filtered_movies:
            selected_movie = st.selectbox("Select Movie", filtered_movies, key="filtered_select")
        else:
            st.warning("No movies found")
            selected_movie = st.selectbox("Select Movie", movies['title'].values, key="all_movies")
    else:
        selected_movie = st.selectbox("Select Movie", movies['title'].values, key="main_select")

# Clear the history selection after it's been used
if 'selected_from_history' in st.session_state:
    del st.session_state.selected_from_history

# Recommendation Slider
num_recs = st.slider(
    "Number of recommendations", 
    min_value=3, 
    max_value=10, 
    value=5,
    help="Choose how many movie recommendations you want"
)

st.markdown(f'<p style="text-align: center; color: #e50914; font-weight: 600; margin: 10px 0;">🎯 Selected: {num_recs} recommendations</p>', unsafe_allow_html=True)

# Action Buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎬 GET RECOMMENDATIONS", type="primary", key="main_recommend"):
        if selected_movie:
            # Add to history
            st.session_state.recommendation_history.append({
                'movie': selected_movie,
                'timestamp': datetime.now().strftime("%H:%M"),
                'count': num_recs
            })
            st.session_state.total_recommendations += 1
            
            # Show loading with progress
            with st.spinner("🎬 Finding perfect matches..."):
                time.sleep(2)  # Simulate processing time
                recommendations, error = get_recommendations(selected_movie, num_recs)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state.current_recommendations = recommendations
                st.session_state.show_recommendations = True
                st.session_state.recommended_for = selected_movie
                st.success("🎉 Perfect matches found!")
                st.rerun()
        else:
            st.warning("⚠️ Please select a movie first!")

with col2:
    if st.button("🎰 Random Pick", key="random_pick"):
        random_movie = random.choice(movies['title'].tolist())
        st.info(f"🎬 Random pick: **{random_movie}**")
        st.session_state.selected_from_history = random_movie
        st.rerun()

with col3:
    if st.button("🎲 Surprise Me", key="surprise_me"):
        random_movie = random.choice(movies['title'].tolist())
        
        # Add to history
        st.session_state.recommendation_history.append({
            'movie': random_movie,
            'timestamp': datetime.now().strftime("%H:%M"),
            'count': num_recs
        })
        st.session_state.total_recommendations += 1
        
        with st.spinner("🎲 Finding surprise recommendations..."):
            time.sleep(2)
            recommendations, error = get_recommendations(random_movie, num_recs)
        
        if not error:
            st.session_state.current_recommendations = recommendations
            st.session_state.show_recommendations = True
            st.session_state.recommended_for = random_movie
            st.success(f"🎲 Surprise pick: **{random_movie}**")
            st.rerun()

with col4:
    if st.button("🗑️ Clear", key="clear_all"):
        st.session_state.show_recommendations = False
        st.session_state.current_recommendations = []
        st.success("🧹 Cleared all results!")
        st.rerun()

# Recommendations Section
if st.session_state.show_recommendations and st.session_state.current_recommendations:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🎬</div>
        <h2 class="section-title">Recommended for You</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show based on movie
    st.markdown(f'<p style="text-align: center; color: #999; margin: 20px 0; font-size: 16px;">Based on: <span style="color: #e50914; font-weight: 700;">{st.session_state.recommended_for}</span></p>', unsafe_allow_html=True)
    
    # Display recommendations in grid
    recommendations = st.session_state.current_recommendations
    
    # Create two columns for grid layout
    cols = st.columns(2)
    for i, rec in enumerate(recommendations):
        with cols[i % 2]:
            st.markdown(f'''
            <div class="movie-card">
                <div class="movie-rank">{i + 1}</div>
                <div class="movie-title">{rec['title']}</div>
                <div class="movie-match">Match: {rec['similarity_score']}%</div>
                <div class="movie-label">⭐ Recommended for you</div>
            </div>
            ''', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-logo">NETFLIX</div>
    <div class="footer-text">Movie Recommendation System</div>
    <div class="footer-subtext">Powered by Advanced Machine Learning • Built with ❤️ using Streamlit</div>
</div>
""", unsafe_allow_html=True)