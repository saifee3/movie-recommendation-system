# 🎥 Movie Recommendation System
<div align="center">   <img src="https://github.com/user-attachments/assets/83013b61-4213-462a-8842-50013c822071" alt="Custom Icon" width="1050" height="400">  </div>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A content and genre-based movie recommender built on the Top Rated TMDB Movies (10K) dataset, featuring a sleek Streamlit interface. Users can pick or search a film and receive personalized suggestions in seconds.

## ✨ Features

- 🎯 Content-based filtering using movie overviews and genres
- 🔍 Search from 10,000+ top-rated TMDB movies
- 📊 Intelligent similarity scoring with TF-IDF and cosine similarity
- 🚀 Instant recommendations via responsive Streamlit interface

## 1. Recommendation Systems

Recommendation systems are information filtering tools designed to predict and present items of interest to users. They analyze patterns within data to suggest products, services, or information that align with individual preferences. These systems have become integral in enhancing user experience across various platforms by personalizing content delivery.

<div align="center">   <img src="https://github.com/user-attachments/assets/fe3f204b-ae5c-4e55-87c8-1028be4dd355" alt="Custom Icon" width="600" height="350">  </div>

***Historical Context and Evolution:***

The concept of recommendation systems dates back to the early days of the internet when the need arose to manage and personalize the vast amounts of available information. Initially, simple algorithms based on user demographics and item popularity were employed. Over time, as data collection and processing capabilities advanced, more sophisticated methods like collaborative and content-based filtering emerged, leading to the complex and efficient systems we encounter today.

***Applications Across Industries:***

- **E-commerce:** Platforms like Amazon utilize them to suggest products based on user browsing and purchase history.
- **Streaming Services:** Netflix and Spotify recommend movies and music tailored to user preferences.
- **Social Media:** Facebook and Twitter suggest friends, pages, or content to users.
- **Online Education:** Coursera and Udemy recommend courses based on user interests and past enrollments.

## 2. Motivation Behind Recommendation Systems

***1. Addressing Information Overload:*** 

In the digital age, users are inundated with vast amounts of information and choices. Recommendation systems mitigate this overload by filtering and presenting options that are most relevant to the user, thereby simplifying decision-making processes.

***2. Enhancing User Experience and Engagement:*** 

By personalizing content, recommendation systems make user interactions more engaging and satisfying. When users receive suggestions that align with their preferences, they are more likely to remain engaged with the platform, leading to increased usage and loyalty.

***3. Driving Business Value and Revenue:*** 

For businesses, effective recommendation systems can lead to increased sales, higher customer retention rates, and a better understanding of consumer behavior. By presenting users with items they are likely to purchase or engage with, companies can boost their revenue and market share.

## 3. Types of Recommendation Systems

***1. Content-Based Filtering:***

This approach recommends items by analyzing the content of items and a user's profile. It assumes that if a user liked an item in the past, they would prefer similar items in the future. For instance, if a user watches science fiction movies, the system will recommend other movies within the same genre. 

<div align="center">   <img src="https://github.com/user-attachments/assets/a28d3707-08b4-40ff-89f6-8b3062aa5781" alt="Custom Icon" width="350" height="350">  </div>

***2. Collaborative Filtering:***

Collaborative filtering methods make recommendations based on the behavior and preferences of similar users. There are two main types:

- ***User-Based Collaborative Filtering:*** Identifies users with similar tastes and recommends items that those users have liked.

- **Item-Based Collaborative Filtering:** Finds items that are similar to those the user has shown interest in and recommends them.

  <div align="center">   <img src="https://github.com/user-attachments/assets/31e50527-8c34-47ab-86cc-f2bd975aff22" alt="Custom Icon" width="450" height="350">  </div>

***3. Hybrid Approaches:***

Hybrid recommendation systems combine content-based and collaborative filtering methods to leverage the strengths of both. By doing so, they aim to provide more accurate and diverse recommendations. 

<div align="center">   <img src="https://github.com/user-attachments/assets/9ae98ebf-a3c2-4bd2-96f1-3aa0dc8e7b5a" alt="Custom Icon" width="700" height="350">  </div>

***4. Knowledge-Based Systems:***

These systems recommend items based on explicit knowledge about item attributes and user preferences. They are often used in domains where user preferences are well-defined, and item attributes are critical, such as recommending financial services or real estate properties.

## 4. Recommendation Models and Architectures

***1. Memory-Based (Neighborhood-Based):***

These methods compute similarities between users or items using the entire dataset. They are straightforward but can suffer from scalability issues as the dataset grows.

***2. Model-Based:***

Utilize machine learning algorithms to learn a predictive model from the data. These models can generalize better to unseen data and handle large datasets more efficiently.

***3. Matrix Factorization Techniques:***

Matrix factorization methods, such as Singular Value Decomposition (SVD), decompose the user-item interaction matrix into latent factors representing users and items. This technique helps in uncovering hidden patterns in the data and is widely used in collaborative filtering.

***4. Deep Learning Models in Recommendations:*** 

Deep learning approaches have been employed to capture complex user-item interactions and nonlinear relationships. Models like neural collaborative filtering and deep matrix factorization have shown promising results in enhancing recommendation accuracy.

***5. Scalability and Real-Time Processing Considerations:*** 

As recommendation systems handle increasing amounts of data, scalability becomes crucial. Techniques such as approximate nearest neighbors and distributed computing frameworks are employed to ensure that recommendations can be generated in real-time without compromising performance.

## 5. Performance Metrics and Evaluation

***1. Accuracy Metrics:***

- **Precision:** The proportion of recommended items that are relevant.
- **Recall:** The proportion of relevant items that are recommended.
- **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.

***2. Ranking Metrics:***

- **Mean Average Precision (MAP):** Evaluates the quality of ranked recommendations by considering the order of relevant items.

- **Normalized Discounted Cumulative Gain (NDCG):** Assesses the usefulness of a recommendation based on the position of relevant items in the list, giving higher scores for relevant items appearing earlier. 

***3. Diversity and Novelty Measures:***

- **Diversity:** Ensures that recommended items are varied and not too similar to each other.
- **Novelty:** Measures how new or unexpected the recommended items are to the user,


## 6. Design Considerations
1. **Cold-Start:** Use content features or prompt initial ratings.
2. **Data Sparsity:** Apply matrix factorization and regularization.
3. **Contextualization:** Incorporate time, location, or session data for dynamic recommendations.
4. **Ethics & Privacy:** Minimize data collection, ensure transparency, and comply with regulations (e.g., GDPR).

**1. Cold Start Problem and Solutions**

**Challenge:**  
The cold start problem occurs when a recommendation system has little or no historical data on new users or items. Without sufficient interactions, the system struggles to generate accurate recommendations.

**Common Solutions:**  
- **Content-Based Approaches:**  
  Leverage item attributes (e.g., metadata, text descriptions) to recommend items that are similar in content to those a new user might like. For example, if a new movie is released, its genre, director, and synopsis can be used to match it with similar items.  
- **Hybrid Models:**  
  Combine collaborative filtering with content-based methods so that even with sparse interaction data, the system can use content cues to provide recommendations.  
- **Demographic-Based Recommendations:**  
  Use demographic information (age, gender, location) to infer initial preferences for new users.  
- **Active Learning:**  
  Prompt new users to rate a few items upon signup. These initial ratings help seed their profile so that the system can start generating recommendations sooner.


**2. Handling Data Sparsity**

**Challenge:**  
Data sparsity is common in user-item interaction matrices, where only a small fraction of all possible user-item pairs have interactions. Sparse data can degrade the performance of collaborative filtering algorithms by making it difficult to detect patterns.

**Common Solutions:**  
- **Matrix Factorization:**  
  Techniques such as Singular Value Decomposition (SVD) or Alternating Least Squares (ALS) reduce the dimensionality of the interaction matrix and reveal latent factors. This approach helps generalize better in sparse settings.  
- **Neighborhood Models with Similarity Thresholds:**  
  Instead of using all available data, limit recommendations to only those neighbors or items that exceed a similarity threshold to avoid noise from sparse interactions.  
- **Data Imputation and Regularization:**  
  Impute missing values with baseline estimates (like global averages) and use regularization during model training to prevent overfitting on the sparse data.


**3. Incorporating Contextual Information**

**Challenge:**  
Traditional recommendation models often assume static preferences, but user interests can vary with time, location, and situation. Contextual information (such as time of day, current trends, or location) can provide additional cues for more relevant recommendations.

**Common Solutions:**  
- **Context-Aware Recommender Systems (CARS):**  
  Extend recommendation models to include contextual variables. For example, a context-aware model might adjust recommendations based on the season, day of the week, or weather.  
- **Session-Based Recommendations:**  
  Analyze the sequence of user interactions within a session to capture short-term interests and generate more dynamic recommendations.  
- **Hybrid Models:**  
  Combine contextual information with traditional user-item interactions to improve personalization. For instance, a recommender could use a weighted combination of a context-free prediction and a context-specific adjustment.

**4. Ethical and Privacy Concerns**

**Challenge:**  
Recommendation systems collect and process personal data, which raises important ethical and privacy issues. Users’ behavioral data, preferences, and sometimes sensitive information are used to drive recommendations.

**Common Solutions and Considerations:**  
- **Data Minimization:**  
  Collect only the data necessary for making effective recommendations. This reduces the risk of exposing sensitive user information.  
- **User Consent and Transparency:**  
  Clearly inform users about what data is collected and how it will be used. Obtain explicit consent and offer options to control data sharing.  
- **Anonymization and Security:**  
  Anonymize data where possible and ensure robust security practices to protect stored user data.  
- **Bias and Fairness:**  
  Monitor for and mitigate biases that may arise in the training data or model. Ensure that recommendations do not inadvertently reinforce stereotypes or discrimination.  
- **Regulatory Compliance:**  
  Ensure that the system complies with data protection regulations such as GDPR or CCPA, which govern how personal data is collected, stored, and processed.

## Sentiment Based Product Recommendation System Steps

<div align="center">   <img src="https://github.com/user-attachments/assets/2d499a4e-ba06-4ce0-b603-25b7eaa4e761" alt="Custom Icon" width="1050" height="400">  </div>

Link to full paper: https://www.researchgate.net/publication/377201036_Sentiment_Based_Product_Recommendation_System_Using_Machine_Learning_Techniques

## Project Implementation
This Movie Recommendation System employs a **content-based filtering** methodology. This approach recommends movies by analyzing the content features of items and matching them to user preferences. In this implementation, the system utilizes Term Frequency-Inverse Document Frequency (**TF-IDF**) vectorization to process movie overviews, transforming them into numerical representations. Additionally, it applies one-hot encoding to represent movie genres. These feature vectors are then combined and the **cosine similarity** metric is used to measure the similarity between movies. When a user selects a movie, the system calculates similarity scores between the chosen movie and others in the dataset, recommending those with the highest similarity scores.

### Dataset

- **Source:** [Top Rated TMDB Movies (10K) on Kaggle](https://www.kaggle.com/datasets/ahsanaseer/top-rated-tmdb-movies-10k)  
- **Key Fields:** `title`, `overview`, `genre`, `vote_average`, `vote_count`, `release_date`

### Feature Engineering

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack

df = pd.read_csv('dataset.csv')
df['overview'] = df['overview'].fillna('')
df['genres_list'] = df['genre'].fillna('').apply(lambda s: s.split(',') if s else [])

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_mat = tfidf.fit_transform(df['overview'])

mlb = MultiLabelBinarizer(sparse_output=True)
genre_mat = mlb.fit_transform(df['genres_list'])

feature_matrix = hstack([tfidf_mat, genre_mat])
```

### Similarity Computation

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(feature_matrix, feature_matrix, dense_output=False)
```

### Recommendation Function

```python
def recommend(title: str, top_n: int = 5) -> list[str]:
    idx = df.index[df['title'] == title][0]
    sims = list(enumerate(cosine_sim[idx].toarray().ravel()))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1: top_n+1]
    return df['title'].iloc[[i for i,_ in sims]].tolist()
```

### Streamlit App

```python
import streamlit as st
from recommender import recommend, df

st.set_page_config(page_title="🎬 TMDB Recommender", layout="wide")
# ... (custom CSS & sidebar code) ...
if st.button("Recommend"):
    recs = recommend(selected_movie, n_rec)
    # render cards with title, rating, year
```

## Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/tmdb-movie-recommender.git
   cd tmdb-movie-recommender
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Download `dataset.csv` into the project root.


## Usage

- **Notebook:** Open `notebook.ipynb` to explore data and test the `recommend()` function.  
- **App:**  
  ```bash
  streamlit run app.py
  ```

## Folder Structure

```
tmdb-movie-recommender/
├── app.py
├── recommender.py
├── dataset.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```

## Dependencies

- streamlit  
- pandas  
- scikit-learn  
- scipy  

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/foo`)  
3. Commit your changes (`git commit -m "Add feature"`)  
4. Push to the branch (`git push origin feature/foo`)  
5. Open a Pull Request

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Credits

- **MIT License Icon:** [Open Source Initiative](https://opensource.org/)
- **Python Icon:** [Python Software Foundation](https://www.python.org/)
- **Streamlit Icon:** [Streamlit](https://streamlit.io/)
- **Scikit-learn Icon:** [scikit-learn](https://scikit-learn.org/)
- Banner Image by **Real Python** on https://realpython.com/build-recommendation-engine-collaborative-filtering/
- Other Images Sources:
  
    Image 1: https://www.researchgate.net/figure/Structure-of-a-recommender-system_fig2_220827211
  
    Image 2: https://www.stratascratch.com/blog/step-by-step-guide-to-building-content-based-filtering/
  
    Image 3: https://medium.com/@ashmi_banerjee/understanding-collaborative-filtering-f1f496c673fd
  
    Image 4: https://www.mdpi.com/2076-3417/13/23/12774
  
  
  
    



