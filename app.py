import streamlit as st
import pandas as pd
import requests
import pickle # Agar aapne model save kiya hai toh, nahi toh direct load karein

# --- 1. Data Load Karein (Sabse Pehle) ---
# Ensure karein ki moovie_name.csv usi folder mein hai jahan app.py hai
df = pd.read_csv('moovie_name.csv') 

# --- 2. Similarity Logic (Yahan load ya calculate karein) ---
# Agar aapne similarity matrix compute kar li hai:
# (Yahan main pichle code ka logic use kar raha hoon)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df['side_genre'] = df['side_genre'].fillna('')
df['Director'] = df['Director'].fillna('')
df['Actors'] = df['Actors'].fillna('')
df['combined_features'] = (df['main_genre'] + " " + df['side_genre'] + 
                          " " + df['Director'] + " " + df['Actors']).str.lower()

cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['combined_features'])
similarity = cosine_similarity(count_matrix)

# --- 3. Functions ---
api_key = "828ef97a068897f3e64365154d09b4a0"

def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    try:
        data = requests.get(url).json()
        poster_path = data['results'][0]['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://www.publicdomainpictures.net/pictures/280000/velka/not-found-image-15383864787lu.jpg"

def recommend(movie):
    movie_index = df[df['Movie_Title'] == movie].index[0] # Dhyaan dein column name 'Movie_Title' hai ya 'Title'
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        title = df.iloc[i[0]].Movie_Title
        recommended_movies.append(title)
        recommended_posters.append(fetch_poster(title))
    return recommended_movies, recommended_posters

# --- 4. Streamlit UI ---
st.title('Movie Recommender System')

# Column check: Aapki CSV mein column ka naam 'Movie_Title' hai
selected_movie = st.selectbox(
    'Movie select karein:',
    df['Movie_Title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie)
    
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(names[idx])
            st.image(posters[idx])