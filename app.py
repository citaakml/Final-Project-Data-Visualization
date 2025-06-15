import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Recommender & Analytics", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df['genres'] = df['genres'].apply(lambda x: ' '.join([i['name'] for i in ast.literal_eval(x)]))
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['overview'] = df['overview'].fillna('')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    return df

df = load_data()

# TF-IDF + cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(title, top_n=5):
    idx = df[df['title'] == title].index
    if idx.empty:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommendations = df.iloc[[i[0] for i in sim_scores]]
    recommendations['similarity'] = [round(i[1], 3) for i in sim_scores]
    return recommendations[['title', 'overview', 'vote_average', 'popularity', 'similarity']]

# App Title & Team Info
st.title("Movie Recommender & Analytics")
st.markdown(
    """
    **Anggota Kelompok:**  
    - Citakamalia (203012320021)  
    - Ihsani Hawa Arsytania (203012320027)  
    - Arliyanna Nilla (203012320035)  
    """
)

# Global Filter
st.sidebar.header("Filter Global")
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
selected_years = st.sidebar.slider("Filter Tahun Rilis", min_year, max_year, (2000, 2015))
genre_options = sorted(set(' '.join(df['genres']).split()))
selected_genres = st.sidebar.multiselect("Filter Genre", genre_options, default=["Action", "Comedy"])

# Apply Global Filter
df_filtered = df[
    (df['release_year'].between(selected_years[0], selected_years[1])) &
    (df['genres'].apply
