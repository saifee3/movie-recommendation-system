import streamlit as st
from recommender import recommend, df

# ─── 1) PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 TMDB Movie Recommender",
    layout="wide",
)

# ─── 2) CUSTOM CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar Gradient ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #FF851B, #CCFF00);  /* orange → electric lime */ :contentReference[oaicite:0]{index=0}
    padding: 1rem;
}

/* Main area background & font */
.reportview-container .main {
    background: #f0f2f6;
    font-family: 'Montserrat', sans-serif;
}

/* Buttons */
div.stButton > button {
    background-color: #FF851B !important;
    color: #FFFFFF !important;
    border-radius: 25px;
    padding: 0.6em 1.2em;
    font-weight: 600;
}

/* Inputs */
.stTextInput>div>div>input,
.stSelectbox>div>div>div>div {
    border: 2px solid #FF851B !important;
    border-radius: 20px;
    padding: 0.5em 1em;
}

/* Recommendation cards */
.recommend-card {
    background: #FFFFFF;
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.recommend-card h4 {
    margin: 0 0 0.5rem;
    color: #001F3F;
    font-weight: 600;
}
.recommend-card p {
    margin: 0;
    color: #555555;
}
</style>
""", unsafe_allow_html=True)


# ─── 3) SIDEBAR: PICK, SEARCH & SLIDER ────────────────────────────────────────
st.sidebar.header("🎬 Pick a Movie")
movie_list = df['title'].tolist()
selected_movie = st.sidebar.selectbox("Choose from all movies", movie_list)

st.sidebar.header("🔎 Search")
search_query = st.sidebar.text_input("Type to search...", "")
if search_query:
    matches = df[df['title'].str.contains(search_query, case=False, na=False)]
    if matches.empty:
        st.sidebar.write("No matches found.")
    else:
        selected_movie = st.sidebar.selectbox(
            "Select from search results",
            matches['title'].tolist()
        )

st.sidebar.header(" Number of Recommendations")
n_rec = st.sidebar.slider("How many?", 1, 10, 5)

# ─── 4) MAIN: BUTTON & RESULTS ────────────────────────────────────────────────
st.header("---Movie Recommendation System---")
if st.button("Recommend"):
    recs = recommend(selected_movie, top_n=n_rec)
    if not recs:
        st.warning("Movie not found in database.")
    else:
        st.subheader(f"Because you liked **{selected_movie}**, you might also like:")
        for i, title in enumerate(recs, 1):
            # Lookup rating & release year
            movie = df[df['title'] == title].iloc[0]
            rating = movie['vote_average']
            year = str(movie['release_date']).split('-')[0]
            st.markdown(
                f"""
                <div class="recommend-card">
                  <h4>{i}. {title}</h4>
                  <p>⭐ {rating} &nbsp;|&nbsp; Release Year: {year}</p>
                </div>
                """,
                unsafe_allow_html=True
            )