import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="TMDB Movie Recommender & Analytics", layout="wide")

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

# Global filter
st.sidebar.header("Filter Global")
selected_years = st.sidebar.slider("Tahun Rilis", int(df['release_year'].min()), int(df['release_year'].max()), (2000, 2015))
genre_options = sorted(set(' '.join(df['genres']).split()))
selected_genres = st.sidebar.multiselect("Genre", genre_options, default=["Action", "Comedy"])

df_filtered = df[
    (df['release_year'].between(selected_years[0], selected_years[1])) &
    (df['genres'].apply(lambda g: any(genre in g for genre in selected_genres)))
]

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

# Title & Info
st.title("TMDB Movie Recommender & Analytics")
st.markdown("""
Citakamalia (203012320021)  
Ihsani Hawa Arsytania (203012320027)  
Arliyanna Nilla (203012320035)
""")

tab1, tab2 = st.tabs(["Rekomendasi Film", "Eksplorasi Visualisasi"])

# Tab 1: Rekomendasi Film
with tab1:
    st.header("Rekomendasi Film Berdasarkan Sinopsis")
    movie_list = df['title'].sort_values().unique()
    selected_movie = st.selectbox("Pilih film favoritmu:", movie_list)
    top_n = st.slider("Jumlah rekomendasi:", 1, 10, 5)

    if st.button("Rekomendasikan"):
        with st.spinner("Mencari film yang mirip..."):
            recs = recommend(selected_movie, top_n)
            if not recs.empty:
                for _, row in recs.iterrows():
                    st.markdown(f"### 🎥 {row['title']}")
                    st.write(f"**Rating:** {row['vote_average']} | **Popularitas:** {round(row['popularity'], 1)} | **Skor Kemiripan:** {row['similarity']}")
                    st.write(row['overview'])
                    st.markdown("---")
            else:
                st.warning("Film tidak ditemukan dalam daftar.")

# Tab 2: Visualisasi
with tab2:
    st.header("Eksplorasi Film Berdasarkan Tahun & Genre")

    # Jumlah Film per Tahun
    st.subheader("Jumlah Film Dirilis per Tahun")
    yearly = df_filtered['release_year'].value_counts().sort_index().reset_index()
    yearly.columns = ['Tahun', 'Jumlah Film']
    chart_year = alt.Chart(yearly).mark_line(point=True).encode(
        x='Tahun:O', y='Jumlah Film:Q', tooltip=['Tahun', 'Jumlah Film']
    ).properties(width=700, height=400)
    st.altair_chart(chart_year)

    # Distribusi Rating
    st.subheader("Distribusi Rating Film")
    chart_rating = alt.Chart(df_filtered).mark_bar().encode(
        x=alt.X('vote_average:Q', bin=alt.Bin(maxbins=20), title="Rating"),
        y='count():Q',
        tooltip=['count()']
    ).properties(width=700, height=400)
    st.altair_chart(chart_rating)

    # Distribusi Durasi
    st.subheader("Distribusi Durasi Film")
    chart_runtime = alt.Chart(df_filtered.dropna(subset=['runtime'])).mark_bar().encode(
        x=alt.X('runtime:Q', bin=alt.Bin(maxbins=30), title="Durasi (menit)"),
        y='count():Q',
        tooltip=['count()']
    ).properties(width=700, height=400)
    st.altair_chart(chart_runtime)

    # Film Terpopuler
    st.subheader("10 Film Paling Populer")
    top_pop = df_filtered.sort_values(by='popularity', ascending=False).head(10)[['title', 'popularity']]
    chart_pop = alt.Chart(top_pop).mark_bar().encode(
        x=alt.X('popularity:Q'),
        y=alt.Y('title:N', sort='-x'),
        tooltip=['title', 'popularity']
    ).properties(width=700, height=400)
    st.altair_chart(chart_pop)

    # Scatter Plot: Budget vs Rating by Genre
    st.subheader("Korelasi Budget vs Rating (warna berdasarkan Genre Utama)")
    scatter_data = df_filtered.dropna(subset=['budget', 'vote_average'])
    scatter_data = scatter_data[scatter_data['budget'] > 0]
    scatter_data['main_genre'] = scatter_data['genres'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Unknown')

    chart_scatter = alt.Chart(scatter_data).mark_circle(size=70, opacity=0.6).encode(
        x=alt.X('budget:Q', title='Budget'),
        y=alt.Y('vote_average:Q', title='Rating'),
        color=alt.Color('main_genre:N', legend=alt.Legend(title="Genre Utama")),
        tooltip=['title', 'budget', 'vote_average', 'main_genre']
    ).interactive().properties(width=700, height=400)
    st.altair_chart(chart_scatter)

    # WordCloud: Berdasarkan overview
    st.subheader("WordCloud Berdasarkan Sinopsis Film")
    all_text = ' '.join(df_filtered['overview'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
