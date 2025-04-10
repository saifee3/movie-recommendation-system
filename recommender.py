import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# 1) Load and preprocess
df = pd.read_csv('dataset.csv')
df['overview'] = df['overview'].fillna('')         
df['genre']    = df['genre'].fillna('')            
# extract list of genres
df['genres_list'] = df['genre'].apply(
    lambda s: [g.strip() for g in s.split(',')] if s else []
)

# 2) Feature engineering
# 2a) TF-IDF on overview
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['overview'])  

# 2b) One-hot encode genres
mlb = MultiLabelBinarizer(sparse_output=True)
genre_matrix = mlb.fit_transform(df['genres_list'])  

# 2c) Combine text + genre features
feature_matrix = hstack([tfidf_matrix, genre_matrix])

# 3) Compute cosine similarity once
cosine_sim = cosine_similarity(feature_matrix, feature_matrix, dense_output=False)  

def recommend(title: str, top_n: int = 5) -> list[str]:
    """
    Return top_n movie titles similar to the given title.
    """
    if title not in df['title'].values:
        return []
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx].toarray().ravel()))
    # sort by similarity score descending, skip itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n+1]
    movie_indices = [i for i, _ in sim_scores]
    return df['title'].iloc[movie_indices].tolist()
