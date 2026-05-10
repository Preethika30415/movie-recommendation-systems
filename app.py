import streamlit as st
import pandas as pd
import requests
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    layout="wide"
)

# --------------------------------
# SESSION STATE
# --------------------------------

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = ""

# --------------------------------
# TMDB API KEY
# --------------------------------

API_KEY = "553a58cacf19acdd781f792f02b7ecf8"

# --------------------------------
# DATABASE CONNECTION
# --------------------------------

conn = sqlite3.connect(
    'users.db',
    check_same_thread=False
)

c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users(
    username TEXT,
    password TEXT
)
''')

conn.commit()

# --------------------------------
# FUNCTIONS
# --------------------------------

def add_user(username, password):

    c.execute(
        'INSERT INTO users(username, password) VALUES (?, ?)',
        (username, password)
    )

    conn.commit()


def login_user(username, password):

    c.execute(
        'SELECT * FROM users WHERE username=? AND password=?',
        (username, password)
    )

    data = c.fetchall()

    return data

# --------------------------------
# FETCH MOVIE POSTER
# --------------------------------

def fetch_poster(movie_name):

    try:

        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={movie_name}"
        )

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        data = response.json()

        if (
            "results" in data and
            len(data["results"]) > 0
        ):

            poster_path = data["results"][0].get(
                "poster_path"
            )

            if poster_path:

                return (
                    "https://image.tmdb.org/t/p/w500/"
                    + poster_path
                )

    except Exception as e:

        print("Poster Error:", e)

    return None

# --------------------------------
# TRAILER FUNCTION
# --------------------------------

def get_trailer(movie_name):

    query = movie_name.replace(" ", "+")

    return (
        "https://www.youtube.com/results?search_query="
        + query
        + "+trailer"
    )

# --------------------------------
# LOAD DATASET
# --------------------------------

movies = pd.read_csv("movies.csv")

movies["overview"] = movies["overview"].fillna("")
movies["genre"] = movies["genre"].fillna("")

movies["combined_features"] = (
    movies["genre"] + " " + movies["overview"]
)

# --------------------------------
# TF-IDF VECTORIZATION
# --------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english"
)

feature_vectors = vectorizer.fit_transform(
    movies["combined_features"]
)

# --------------------------------
# COSINE SIMILARITY
# --------------------------------

similarity = cosine_similarity(
    feature_vectors
)

# --------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------

def recommend(movie_name):

    movie_name = movie_name.lower()

    matching_movies = movies[
        movies["title"]
        .str.lower()
        .str.contains(movie_name)
    ]

    if matching_movies.empty:

        return []

    index = matching_movies.index[0]

    similarity_scores = list(
        enumerate(similarity[index])
    )

    sorted_movies = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movies = []

    for movie in sorted_movies[1:6]:

        movie_index = movie[0]

        recommended_movies.append(
            (
                movies.iloc[movie_index]["title"],
                movie[1]
            )
        )

    return recommended_movies

# --------------------------------
# TITLE
# --------------------------------

st.title("🎬 AI Movie Recommendation System")

# --------------------------------
# SIDEBAR MENU
# --------------------------------

menu = ["Login", "Signup"]

choice = st.sidebar.selectbox(
    "Menu",
    menu
)

# --------------------------------
# SIGNUP
# --------------------------------

if choice == "Signup":

    st.subheader("Create Account")

    new_user = st.text_input(
        "Username"
    )

    new_password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Signup"):

        add_user(
            new_user,
            new_password
        )

        st.success(
            "Account Created Successfully"
        )

# --------------------------------
# LOGIN
# --------------------------------

elif choice == "Login":

    st.subheader("Login")

    username = st.text_input(
        "Username",
        key="username"
    )

    password = st.text_input(
        "Password",
        type="password",
        key="password"
    )

    if st.button("Login"):

        result = login_user(
            username,
            password
        )

        if result:

            st.session_state.logged_in = True

            st.success(
                "Logged In Successfully"
            )

        else:

            st.error(
                "Invalid Credentials"
            )

# --------------------------------
# AFTER LOGIN
# --------------------------------

if st.session_state.logged_in:

    st.subheader(
        "Movie Recommendation"
    )

    # ----------------------------
    # WATCHLIST SIDEBAR
    # ----------------------------

    st.sidebar.subheader(
        "My Watchlist"
    )

    if len(st.session_state.watchlist) == 0:

        st.sidebar.write(
            "No movies added yet"
        )

    else:

        for item in st.session_state.watchlist:

            st.sidebar.write(item)

    # ----------------------------
    # CLEAR WATCHLIST
    # ----------------------------

    if st.sidebar.button(
        "Clear Watchlist"
    ):

        st.session_state.watchlist = []

        st.rerun()

    # ----------------------------
    # MOVIE SEARCH
    # ----------------------------

    selected_movie = st.text_input(
        "Enter Movie Name",
        value=st.session_state.selected_movie
    )

    # ----------------------------
    # RECOMMEND BUTTON
    # ----------------------------

    if st.button("Recommend"):

        st.session_state.selected_movie = (
            selected_movie
        )

        st.session_state.recommendations = recommend(
            selected_movie
        )

    # ----------------------------
    # SHOW RECOMMENDATIONS
    # ----------------------------

    if st.session_state.recommendations:

        st.subheader(
            "Recommended Movies"
        )

        for movie, score in st.session_state.recommendations:

            st.write("##", movie)

            # ------------------------
            # RATING
            # ------------------------

            movie_data = movies[
                movies["title"] == movie
            ]

            if not movie_data.empty:

                rating = movie_data[
                    "rating"
                ].values[0]

                st.write(
                    f"⭐ IMDb Rating: {rating}"
                )

            else:

                st.write(
                    "⭐ IMDb Rating: Not Available"
                )

            # ------------------------
            # SIMILARITY SCORE
            # ------------------------

            st.write(
                f"Similarity Score: {round(score * 100, 2)}%"
            )

            # ------------------------
            # POSTER
            # ------------------------

            poster = fetch_poster(movie)

            if poster:

                st.image(
                    poster,
                    width=250
                )

            else:

                st.write(
                    "Poster Not Available"
                )

            # ------------------------
            # TRAILER
            # ------------------------

            trailer = get_trailer(movie)

            st.markdown(
                f"[▶ Watch Trailer]({trailer})"
            )

            # ------------------------
            # WATCHLIST BUTTON
            # ------------------------

            if st.button(
                f"Add {movie} to Watchlist",
                key=f"watch_{movie}"
            ):

                if movie not in st.session_state.watchlist:

                    st.session_state.watchlist.append(
                        movie
                    )

                    st.success(
                        "Added to Watchlist"
                    )

                else:

                    st.warning(
                        "Already in Watchlist"
                    )

    else:

        if st.session_state.selected_movie != "":

            st.error(
                "Movie Not Found"
            )

    # --------------------------------
    # LOGOUT BUTTON
    # --------------------------------

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False

        st.session_state.watchlist = []

        st.session_state.recommendations = []

        st.session_state.selected_movie = ""

        if "username" in st.session_state:
            st.session_state.username = ""

        if "password" in st.session_state:
            st.session_state.password = ""

        st.rerun()