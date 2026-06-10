# RecFlix — AI-Powered Entertainment Recommendation System

RecFlix is a full-stack web application that recommends movies, books, and TV shows using a hybrid recommendation engine combining content-based filtering and collaborative filtering. Built with Django and Python, it delivers personalized entertainment suggestions across three media types in a modern Netflix-inspired interface.

---

## Features

- **Cross-media recommendations** — Discover movies related to books you love, or TV shows similar to movies you've watched
- **Personalized homepage** — AI-powered picks tailored to your individual taste based on your rating history
- **Genre browsing** — Browse content by genre across all three media types
- **Rating system** — Rate movies, books, and TV shows on a 1–5 star scale to improve your recommendations
- **User authentication** — Secure signup, login, and logout system
- **Responsive dark UI** — Netflix-inspired design with smooth hover animations and scrollable genre rows

---

## How the Recommendation System Works

RecFlix uses a **hybrid recommendation approach** combining two distinct algorithms that work together to deliver accurate and personalized suggestions.

### 1. Content-Based Filtering

Content-based filtering powers the **"Similar Items"** section on every detail page. It recommends items that are textually and thematically similar to the one you're currently viewing.

**How it works step by step:**

**Step 1 — Feature Extraction**
For each movie, book, and TV show, the system combines key descriptive fields into a single text string:
- Movies: title + plot + genre + actors + director
- Books: title + author + plot + genre
- TV Shows: title + plot + genre

**Step 2 — Keyword Extraction with RAKE**
The system uses the **Rapid Automatic Keyword Extraction (RAKE)** algorithm from the `rake-nltk` library to extract the most significant keywords from each item's combined text. RAKE identifies keyword phrases by analysing word co-occurrence and frequency, filtering out stopwords to focus on meaningful terms.

**Step 3 — Bag of Words Construction**
All extracted keywords are combined into a single "bag of words" string per item, representing its thematic fingerprint.

**Step 4 — TF-IDF Vectorisation**
The bag of words strings are transformed into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)** from scikit-learn. TF-IDF weighs words by how important they are to a specific item relative to all items in the dataset — common words across all items get lower weight while distinctive words get higher weight.

**Step 5 — Cosine Similarity**
The system computes the **cosine similarity** between the TF-IDF vector of the selected item and every other item in the combined dataset (movies + books + TV shows). Cosine similarity measures the angle between two vectors — a score of 1 means identical, 0 means completely unrelated.

**Step 6 — Cross-Media Recommendations**
The similarity scores are ranked and the top results are returned. Because all three media types are vectorised together in one matrix, the system naturally surfaces cross-media recommendations — a book about space exploration might surface alongside a sci-fi movie, even though they are different media types.

---

### 2. Collaborative Filtering (SVD)

Collaborative filtering powers the **"Picked For You"** section on every homepage. Unlike content-based filtering which looks at item descriptions, collaborative filtering looks at **user behaviour patterns**.

**How it works step by step:**

**Step 1 — Ratings Matrix**
The system loads all user ratings from CSV files (one per media type). Each rating is a triplet of (username, item_id, rating).

**Step 2 — Matrix Factorisation with SVD**
RecFlix uses **Singular Value Decomposition (SVD)** from the `scikit-surprise` library — the same algorithm that won the Netflix Prize competition. SVD decomposes the user-item ratings matrix into latent factor matrices, capturing hidden patterns like:
- Users who liked dark psychological thrillers
- Users who prefer slow-burn character dramas
- Users who enjoy sci-fi across all three media types

**Step 3 — Rating Prediction**
Once the SVD model is trained, it can predict what rating any user would give to any item they haven't rated yet. The formula estimates a score based on:
- The user's general rating tendencies (user bias)
- The item's overall popularity (item bias)
- The interaction between the user's latent preferences and the item's latent features

**Step 4 — Personalised Ranking**
All unrated items are scored for the current user. The top 12 highest predicted scores are surfaced as the "Picked For You" recommendations, excluding items the user has already rated.

**Step 5 — Cross-Validation**
The SVD model runs 5-fold cross-validation on each request using RMSE (Root Mean Squared Error) as the evaluation metric to ensure prediction quality.

---

### 3. Hybrid Cross-Media Engine

The most distinctive feature of RecFlix is its **cross-media recommendation capability**. When you open a movie detail page, the system doesn't just recommend other movies — it also surfaces books and TV shows with similar themes, plots, and genres.

This works because the content-based pipeline processes all three datasets simultaneously:

```
Books Dataset ──┐
                ├──► Combined TF-IDF Matrix ──► Cosine Similarity ──► Cross-Media Results
Movies Dataset ─┤
                │
TV Shows Dataset┘
```

Items are identified by prefixed IDs (`m_` for movies, `b_` for books, `t_` for TV shows), allowing the system to route recommendations back to the correct detail pages.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 3.2, Python 3.12 |
| Recommendation | scikit-learn, scikit-surprise, rake-nltk, nltk |
| Data Processing | pandas, numpy, scipy |
| Database | SQLite3 |
| Frontend | HTML5, CSS3 (custom dark theme) |
| Fonts | Google Fonts — Bebas Neue, DM Sans |

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone <repo-url>
cd recommendation-system
```

**2. Install dependencies**
```bash
pip install django==3.2.25
pip install pandas numpy scipy
pip install scikit-learn
pip install scikit-surprise
pip install rake-nltk nltk
pip install requests
```

**3. Download NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

**4. Run migrations**
```bash
python manage.py migrate
```

**5. Start the server**
```bash
python manage.py runserver
```

**6. Open in browser**
```
http://127.0.0.1:8000/
```

---

## Project Structure

```
recflix/
├── accounts/        # User authentication (login, signup, logout)
├── movies/          # Movie listings, details, recommendations
├── books/           # Book listings, details, recommendations
├── tvshows/         # TV show listings, details, recommendations
├── api/             # API endpoints for rating submission
├── recflix/         # Project settings and URL configuration
├── datasets/        # CSV files for movies, books, shows and ratings
├── static/          # Static assets (images, icons)
└── manage.py
```

---

## Datasets

RecFlix uses three curated CSV datasets:

| Dataset | Fields |
|---|---|
| `movie_data.csv` | movie_id, movie_title, movie_genre, actors, director, movie_plot, imdb_rating, movie_link |
| `book_data.csv` | book_id, book_title, book_author, book_genre, book_plot, book_rating, book_link |
| `tvshow_data.csv` | show_id, show_name, show_genre, show_plot, show_rating, show_link |
| `movie_ratings.csv` | username, movie_id, rating |
| `book_ratings.csv` | username, book_id, rating |
| `tvshow_ratings.csv` | username, show_id, rating |

---

## Recommendation Algorithm Summary

| Feature | Algorithm | Library |
|---|---|---|
| Similar Movies | TF-IDF + Cosine Similarity | scikit-learn |
| Similar Books | TF-IDF + Cosine Similarity | scikit-learn |
| Similar TV Shows | TF-IDF + Cosine Similarity | scikit-learn |
| Cross-Media | Combined TF-IDF Matrix | scikit-learn |
| Keyword Extraction | RAKE | rake-nltk |
| Personalised Picks | SVD Matrix Factorisation | scikit-surprise |
| Model Evaluation | 5-Fold Cross-Validation RMSE | scikit-surprise |

---

## 👥 Usage

1. Create an account on the signup page
2. Browse movies, books, or TV shows from the homepage
3. Click any item to view its detail page
4. Rate items using the 1–5 star system to train your personalised recommendations
5. Return to the homepage to see your "Picked For You" section update based on your ratings
6. Explore cross-media recommendations on detail pages to discover books related to movies you love

---
