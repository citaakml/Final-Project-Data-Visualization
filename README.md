# Movie Recommender & Analytics

Aplikasi interaktif berbasis **Streamlit** yang menyediakan sistem rekomendasi film dan visualisasi tren dari **TMDB 5000 Movie Dataset** dari tahun 1916-2017. Pengguna dapat mencari film serupa berdasarkan judul film, maka akan memunculkan deskripsi sinopsis, serta mengeksplorasi tren genre dan jumlah perilisan film per tahun.

## Fitur Utama

- **Sistem Rekomendasi Film**  
  Berdasarkan kemiripan sinopsis menggunakan TF-IDF dan cosine similarity.

- **Visualisasi Tren Film**  
  - Genre paling populer
  - Jumlah film dirilis per tahun
  - Tren rating atau popularitas per dekade
  - Tren jumlah film yang dirilis per tahun
  - Distribusi rating film
  - Distribusi durasi film
  - 10 film paling populer
  - WordCloud dari sinopsis/overview film
  - Scatter Plot antara budget film dan rating
  - Filter Interaktif Global berdasarkan tahun rilis dan genre
    
- **Antarmuka Interaktif**  
  Dibangun dengan Streamlit dan mendukung pemfilteran serta analisis eksploratif.

  ## Dataset

Dataset yang digunakan:  
🔗 [TMDB 5000 Movie Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

File yang digunakan: `tmdb_5000_movies.csv` 

1. **Clone repositori ini**
   ```bash
   git clone https://github.com/username-kamu/nama-repo.git
   cd nama-repo

   Install dependensi

2. **Install Dependency**
   ```bash
   pip install -r requirements.txt

4. **Jalankan Aplikasi Streamlit**
   ```bash
   streamlit run app.py
