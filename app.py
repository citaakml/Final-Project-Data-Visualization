import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt

st.set_page_config(page_title="TMDB Movie Recommender", layout="wide")

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

# Streamlit Interface
st.title("🎬 Movie Recommender & Analytics")
st.markdown("""
Citakamalia (203012320021)  
Ihsani Hawa Arsytania (203012320027)  
Arliyanna Nilla (203012320035)
""")

tab1, tab2 = st.tabs(["🔍 Rekomendasi Film", "📊 Eksplorasi & Visualisasi"])

# Tab 1: Rekomendasi
with tab1:
    st.header("Rekomendasi Film Berdasarkan Sinopsis")
    movie_list = df_
