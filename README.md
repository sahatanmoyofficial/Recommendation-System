# 📚 End-to-End Books Recommender System

> **Collaborative Filtering that finds your next favourite book — powered by K-Nearest Neighbours, a 4-stage ML pipeline, and a Streamlit web interface**
>
> Users select any book from a dropdown, click **Show Recommendation**, and instantly see 5 similar books with their cover images — all served from a trained KNN model built on real user ratings data.

---

<div align="center">

[![Python 3.7](https://img.shields.io/badge/Python-3.7.10-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Model-KNN%20NearestNeighbors-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Deployed-orange?logo=amazonaws)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📊 Project Slides

> **Want the visual overview first?** The slide deck covers everything — problem, architecture, pipeline, model logic, and UI — in 12 slides.

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1z21LueUyGZrMeoPwHCWC813cYpajP-Iv/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Business Problem](#1-business-problem) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Data & Features](#6-data--features) |
| 7 | [ML Pipeline — Step by Step](#7-ml-pipeline--step-by-step) |
| 8 | [How the Recommender Works](#8-how-the-recommender-works) |
| 9 | [Streamlit Web Application](#9-streamlit-web-application) |
| 10 | [How to Replicate — Full Setup Guide](#10-how-to-replicate--full-setup-guide) |
| 11 | [Running the Application](#11-running-the-application) |
| 12 | [CI/CD & Cloud Deployment](#12-cicd--cloud-deployment) |
| 13 | [Business Applications & Other Domains](#13-business-applications--other-domains) |
| 14 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 15 | [Troubleshooting](#15-troubleshooting) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Business Problem

### What problem are we solving?

The global books market contains millions of titles. A reader who just finished a novel has no efficient way to discover which of the millions of other books they are most likely to enjoy. Manual recommendations are slow and don't scale; keyword search requires knowing what to look for; bestseller lists reflect popularity, not personal taste.

Core pain points:

- 📚 **Discovery paralysis** — too many books, no personalised signal to navigate them
- 🤷 **Poor generic recommendations** — bestseller lists reflect the crowd, not the individual
- 🔁 **Reader churn** — readers who can't find their next book simply stop reading (and stop buying)
- 💼 **Business impact** — for publishers, booksellers, and libraries, poor discoverability means lost engagement and revenue

### What does this system answer?

> *"Given a book a reader already loves, which 5 books are most similar — based on how real readers rated them?"*

This is **item-based collaborative filtering**: instead of building a profile of the user, we find books that attracted similar rating patterns from the same community of readers.

### Objectives

1. Build a collaborative filtering recommender using K-Nearest Neighbours on the BX-Books dataset
2. Create a clean, validated, user-rating-filtered dataset pipeline
3. Construct a book–user pivot matrix and convert it to a sparse matrix for efficient KNN search
4. Serve recommendations with book cover images via a Streamlit web UI
5. Containerise the application with Docker and deploy to AWS EC2

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | BX-Books dataset (Book-Crossing community ratings) |
| **Data files** | `BX-Books.csv` — book metadata; `BX-Book-Ratings.csv` — user ratings |
| **Filtering rules** | Keep users with 200+ ratings; keep books with 50+ ratings |
| **Recommendation type** | Item-based collaborative filtering |
| **Model** | `sklearn.neighbors.NearestNeighbors(algorithm='brute')` |
| **Similarity** | Cosine distance on sparse book–user pivot matrix |
| **Recommendations** | 5 similar books (6 neighbours, index 0 = the query book itself) |
| **Output** | Book titles + cover image URLs displayed in Streamlit columns |
| **UI Framework** | Streamlit (port 8501) |
| **Serialised artifacts** | `model.pkl`, `book_pivot.pkl`, `final_rating.pkl`, `book_names.pkl` |
| **Deployment** | Docker → AWS EC2 (port 8501) |
| **Python version** | 3.7.10 |

---

## 3. Tech Stack

### Complete Technology Map

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.7.10 | Core language across entire project |
| **ML / Recommender** | Scikit-learn `NearestNeighbors` | KNN model for item similarity search |
| **Sparse Matrix** | SciPy `csr_matrix` | Converts dense pivot table to sparse format for efficient KNN |
| **Data Processing** | Pandas | CSV loading, merging, filtering, pivot table creation |
| **Data Processing** | NumPy | Array operations for index lookups |
| **Serialisation** | Pickle | Saves/loads model, pivot table, book names, final ratings |
| **Config Management** | PyYAML | Reads `config.yaml` for all file paths and settings |
| **Web UI** | Streamlit | Interactive dropdown + image display, training trigger button |
| **Containerisation** | Docker (`python:3.7-slim-buster`) | Packages app; ENTRYPOINT runs Streamlit on port 8501 |
| **Cloud** | AWS EC2 (Ubuntu) | Hosts Streamlit app as Docker container |
| **Logging** | Python `logging` | Timestamped log files in `logs/` directory |
| **Error Handling** | Custom `AppException` | Captures file name + line number for every raised exception |
| **Package** | `setuptools` (`setup.py`) | Installs `books_recommender` as an editable package |
| **Config entities** | `collections.namedtuple` | Immutable typed config containers per pipeline stage |

---

## 4. High-Level Architecture

### System Context

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│                                                                      │
│  [ books_data.zip ]  ──►  [ BX-Books.csv + BX-Book-Ratings.csv ]   │
│       (local ZIP)                  (ingested_data/)                  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        PIPELINE LAYER  (main.py)                     │
│                                                                      │
│  [Stage 0]        [Stage 1]          [Stage 2]          [Stage 3]   │
│  Data Ingest  →  Data Validate  →  Data Transform  →  Model Train   │
│      │               │                  │                  │        │
│      ▼               ▼                  ▼                  ▼        │
│  raw ZIP        clean_data.csv     book_pivot.pkl      model.pkl    │
│  extracted      final_rating.pkl   book_names.pkl                   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER  (app.py)                       │
│                                                                      │
│  [ Streamlit UI ]  port 8501                                         │
│     - Dropdown: select any book title                                │
│     - "Show Recommendation" → KNN query on book_pivot               │
│     - Fetch 5 cover images from final_rating image_url              │
│     - Display: 5 columns × (title + cover image)                    │
│     - "Train Recommender System" → triggers TrainingPipeline        │
│                                                                      │
│  [ Docker Container ]  ←──  [ AWS EC2 ]                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Data & Config Flow Summary

| # | Stage | Input | Key Output |
|---|-------|-------|-----------|
| 0 | **Data Ingestion** | `books_data.zip` (local) | `artifacts/dataset/ingested_data/BX-Books.csv` + `BX-Book-Ratings.csv` |
| 1 | **Data Validation** | Both CSVs | `clean_data.csv`, `final_rating.pkl` |
| 2 | **Data Transformation** | `clean_data.csv` | `book_pivot.pkl`, `book_names.pkl`, `transformed_data.pkl` |
| 3 | **Model Training** | `transformed_data.pkl` | `artifacts/trained_model/model.pkl` |
| 4 | **Serving** | `model.pkl` + `book_pivot.pkl` + `final_rating.pkl` | 5 recommended titles + cover images |

---

## 5. Repository Structure

```
Recommendation-System/
│
├── books_recommender/                      # Core Python package
│   ├── __init__.py
│   ├── components/                         # Pipeline stage implementations
│   │   ├── stage_00_data_ingestion.py      # Download & extract ZIP → CSV files
│   │   ├── stage_01_data_validation.py     # Filter + merge + clean → serialised artifacts
│   │   ├── stage_02_data_transformation.py # Pivot table → sparse-ready PKLs
│   │   └── stage_03_model_trainer.py       # KNN fit → model.pkl
│   ├── config/
│   │   └── configuration.py               # AppConfiguration — builds typed configs from YAML
│   ├── constant/
│   │   └── __init__.py                    # CONFIG_FILE_PATH constant
│   ├── entity/
│   │   └── config_entity.py              # namedtuples: DataIngestionConfig, ModelTrainerConfig …
│   ├── exception/
│   │   └── exception_handler.py          # AppException — captures file + line number
│   ├── logger/
│   │   └── log.py                        # Timestamped log file in logs/
│   ├── pipeline/
│   │   └── training_pipeline.py          # TrainingPipeline — chains all 4 stages
│   └── utils/
│       └── util.py                       # read_yaml_file helper
│
├── config/
│   └── config.yaml                        # All file paths, dirs, URLs
│
├── artifacts/                             # Pipeline outputs (auto-created, gitignored)
│   ├── dataset/
│   │   ├── raw_data/                      # Downloaded ZIP
│   │   ├── ingested_data/                 # Extracted CSVs
│   │   ├── clean_data/                    # clean_data.csv
│   │   └── transformed_data/              # transformed_data.pkl (book pivot)
│   ├── serialized_objects/                # book_names.pkl, book_pivot.pkl, final_rating.pkl
│   └── trained_model/                     # model.pkl (NearestNeighbors)
│
├── templates/
│   └── book_names.pkl                     # Pre-built book names for dropdown (avoids cold start)
│
├── logs/                                  # Timestamped execution logs
│
├── app.py                                 # Streamlit UI — recommendations + training trigger
├── main.py                                # Programmatic training entry point
├── Dockerfile                             # python:3.7-slim-buster, EXPOSE 8501, Streamlit ENTRYPOINT
├── requirements.txt                       # scikit-learn, pandas, numpy, PyYAML, streamlit
├── setup.py                               # Package: books_recommender
└── template.py                            # Project scaffold generator
```

---

## 6. Data & Features

### Dataset: BX-Books (Book-Crossing)

The BX-Books dataset was collected from the Book-Crossing online community. It contains explicit and implicit book ratings from a large number of real users.

| File | Contents | Format |
|------|---------|--------|
| `BX-Books.csv` | Book metadata | ISBN, title, author, year, publisher, image URLs |
| `BX-Book-Ratings.csv` | User ratings | User-ID, ISBN, Book-Rating (0–10) |

### Filtering Pipeline (Stage 1)

Raw ratings data is very sparse — most users have only rated a handful of books. The pipeline applies two quality filters before building the recommendation model:

**Filter 1 — Active users only:**
```python
x = ratings['user_id'].value_counts() > 200
y = x[x].index
ratings = ratings[ratings['user_id'].isin(y)]
```
Keeps only users who have rated more than 200 books — these are engaged community members whose taste signals are meaningful.

**Filter 2 — Popular books only:**
```python
final_rating = final_rating[final_rating['num_of_rating'] >= 50]
```
Keeps only books that have received at least 50 ratings — ensures the pivot matrix has enough signal per book row.

**Filter 3 — Deduplicate:**
```python
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)
```
Removes any duplicate (user, book) rating pairs.

### Book Metadata Used

After merging, the following columns are available per book:

| Column | Description |
|--------|------------|
| `ISBN` | International Standard Book Number |
| `title` | Book title (used as index in pivot table and KNN lookup key) |
| `author` | Book author name |
| `year` | Year of publication |
| `publisher` | Publisher name |
| `image_url` | Large cover image URL (from `Image-URL-L`) — used for poster display in Streamlit |
| `rating` | User's explicit rating (0–10) |
| `user_id` | Anonymised user identifier |

---

## 7. ML Pipeline — Step by Step

The full pipeline is triggered by `python main.py`, which instantiates `TrainingPipeline` and calls `start_training_pipeline()`. Each stage is a self-contained component class with an `initiate_*` method.

---

### Stage 0 — Data Ingestion

**Component:** `stage_00_data_ingestion.py`
**Config entity:** `DataIngestionConfig`

1. Reads `dataset_download_url` from `config.yaml` — currently set to local `books_data.zip`
2. Creates `artifacts/dataset/raw_data/` directory
3. Downloads (or copies) the ZIP file to `raw_data/`
4. Extracts ZIP contents to `artifacts/dataset/ingested_data/`
5. Resulting files: `BX-Books.csv` and `BX-Book-Ratings.csv`

> **Note:** The URL in `config.yaml` currently points to a local file path (`books_data.zip`). For production, this should be updated to an S3 URL or a public download link.

---

### Stage 1 — Data Validation

**Component:** `stage_01_data_validation.py`
**Config entity:** `DataValidationConfig`

Despite the "validation" name, this stage performs the core **data cleaning and preprocessing**:

1. Reads both CSVs with `sep=";"` and `encoding='latin-1'` (BX dataset uses semicolon delimiters)
2. Selects relevant columns from books: `ISBN`, `title`, `author`, `year`, `publisher`, `image_url`
3. Renames messy column headers to clean snake_case
4. Filters to active users (200+ ratings)
5. Merges ratings with books on `ISBN`
6. Counts ratings per book, keeps books with 50+ ratings
7. Drops duplicate (user, book) pairs
8. Saves `clean_data.csv` and serialises `final_rating.pkl`

---

### Stage 2 — Data Transformation

**Component:** `stage_02_data_transformation.py`
**Config entity:** `DataTransformationConfig`

1. Loads `clean_data.csv`
2. Creates a **pivot table**: rows = book titles, columns = user IDs, values = ratings
3. Fills all NaN values with `0` (unrated = 0)
4. Saves three serialised artifacts:
   - `transformed_data.pkl` — the full pivot table (used for model training)
   - `book_names.pkl` — the index of book titles (used to populate Streamlit dropdown)
   - `book_pivot.pkl` — the pivot table again (used by recommendation engine at inference)

---

### Stage 3 — Model Training

**Component:** `stage_03_model_trainer.py`
**Config entity:** `ModelTrainerConfig`

1. Loads `transformed_data.pkl` (book–user pivot table)
2. Converts to a **CSR sparse matrix** via `csr_matrix(book_pivot)` — efficient for high-dimensional sparse data
3. Trains `NearestNeighbors(algorithm='brute')` on the sparse matrix
4. Saves trained model as `artifacts/trained_model/model.pkl`

**Why `algorithm='brute'`?** For sparse high-dimensional data (each book is a vector across thousands of users), brute-force exhaustive search is often faster than tree-based approaches like ball-tree or kd-tree, which degrade in high dimensions.

---

## 8. How the Recommender Works

### Collaborative Filtering — Item-Based

This system uses **item-based collaborative filtering**: it finds books that are "close" to each other in the space of user ratings, not based on content (genre, author, keywords) but on behavioural similarity — the books that the same community of readers tended to rate together.

### The KNN Recommendation Flow

```
User selects: "The Da Vinci Code"
         │
         ▼
book_id = np.where(book_pivot.index == "The Da Vinci Code")[0][0]
         │
         ▼
book_vector = book_pivot.iloc[book_id, :].values.reshape(1, -1)
         │  (1 × N_users sparse vector of ratings for this book)
         ▼
distance, suggestion = model.kneighbors(book_vector, n_neighbors=6)
         │  (6 nearest neighbours: index 0 = the book itself)
         ▼
Fetch titles: book_pivot.index[suggestion[i]]  for i in 1..5
         │
         ▼
Fetch covers: final_rating[final_rating['title'] == name]['image_url']
         │
         ▼
Display: 5 columns in Streamlit — title + cover image
```

### Why 6 Neighbours?

`model.kneighbors(..., n_neighbors=6)` returns 6 results. The **first result (index 0) is always the query book itself** (distance = 0). The app therefore displays results `[1]` through `[5]` — giving exactly 5 distinct recommendations.

### Similarity Metric

`NearestNeighbors` with `algorithm='brute'` uses **cosine distance** by default on the sparse matrix. Two books are "close" if their rating vectors (across thousands of users) point in a similar direction — meaning the same community of readers rated both books similarly.

---

## 9. Streamlit Web Application

The Streamlit app (`app.py`) provides the full user experience — no API knowledge needed.

### Interface Elements

| Element | Description |
|---------|------------|
| **Header** | "End to End Books Recommender System" |
| **Subtext** | Collaborative filtering description |
| **Train button** | `st.button('Train Recommender System')` — triggers full `TrainingPipeline` |
| **Dropdown** | `st.selectbox(...)` — populated from `templates/book_names.pkl` |
| **Show Recommendation button** | Triggers `recommendations_engine(selected_books)` |
| **5-column layout** | Each column: book title (`st.text`) + cover image (`st.image`) |

### The `Recommendation` Class (app.py)

| Method | Purpose |
|--------|---------|
| `recommend_book(book_name)` | KNN lookup → returns 6 book titles + triggers poster fetch |
| `fetch_poster(suggestion)` | Looks up image URLs from `final_rating.pkl` → returns list of cover URLs |
| `recommendations_engine(selected_books)` | Combines both + renders the 5-column Streamlit layout |
| `train_engine()` | Instantiates `TrainingPipeline` and calls `start_training_pipeline()` |

### Cold Start Handling

The dropdown is pre-populated from `templates/book_names.pkl` — a pre-built artifact committed to the repo. This means the app can display the book selector **without running the training pipeline first**, useful for demos and deployments where the full dataset may not be immediately available.

---

## 10. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.7.10
- Git
- Conda (recommended) or `venv`
- Docker Desktop (optional, for containerised testing)
- AWS account (optional, for cloud deployment)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sahatanmoyofficial/Recommendation-System.git
cd Recommendation-System
```

---

### Step 2 — Set Up Python Environment

```bash
# Conda (recommended — Python version matters for pickle compatibility)
conda create -n books python=3.7.10 -y
conda activate books

# Or venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

> ⚠️ **Important:** Use Python 3.7.10 exactly. Pickle files are tied to the Python version — mismatches cause `UnpicklingError`.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# Installs: scikit-learn, pandas, numpy, PyYAML, streamlit
# Also installs books_recommender package in editable mode (-e .)
```

---

### Step 4 — Verify the Data File

The training pipeline expects `books_data.zip` to be present in the project root (the `config.yaml` `dataset_download_url` points to this local file by default). Confirm:

```bash
ls books_data.zip   # Should exist — it ships with the repository
```

If the ZIP is missing, update `config.yaml` to point to a hosted URL:
```yaml
data_ingestion_config:
  dataset_download_url: https://your-s3-bucket.s3.amazonaws.com/books_data.zip
```

---

### Step 5 — Run the Full Training Pipeline

```bash
python main.py
```

This executes all 4 stages in order. After completion, verify:

```bash
ls artifacts/dataset/ingested_data/    # BX-Books.csv, BX-Book-Ratings.csv
ls artifacts/dataset/clean_data/       # clean_data.csv
ls artifacts/serialized_objects/       # final_rating.pkl, book_names.pkl, book_pivot.pkl
ls artifacts/trained_model/            # model.pkl
```

---

### Step 6 — Launch the Streamlit App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

1. Select any book from the dropdown
2. Click **Show Recommendation** — see 5 titles with cover images
3. Click **Train Recommender System** to retrain on the fly

---

## 11. Running the Application

### Local Run

```bash
streamlit run app.py
# http://localhost:8501
```

### Docker Run

```bash
# Build
docker build -t books-recommender:latest .

# Run
docker run -d -p 8501:8501 books-recommender:latest

# Test
curl http://localhost:8501/
```

### Programmatic Training

```bash
python main.py
# Runs all 4 stages, outputs logs to logs/log_<timestamp>.log
```

---

## 12. CI/CD & Cloud Deployment

The project uses a **manual Docker deployment** pattern to AWS EC2 (no GitHub Actions CI/CD configured — this is an improvement opportunity; see Section 14).

### EC2 Manual Deployment Steps

```bash
# 1. On your EC2 instance (Ubuntu) — install Docker
sudo apt-get update -y && sudo apt-get upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# 2. Clone the repository
git clone https://github.com/sahatanmoyofficial/Recommendation-System.git
cd Recommendation-System

# 3. Build and run
docker build -t books-recommender:latest .
docker run -d -p 8501:8501 books-recommender:latest
```

App is then accessible at `http://<EC2-PUBLIC-IP>:8501`

> ⚠️ **EC2 setup note:** Open port **8501** (not 80) in the EC2 Security Group inbound rules.

### Docker Hub Push (Optional)

```bash
docker login
docker tag books-recommender:latest your-dockerhub-username/books-recommender:latest
docker push your-dockerhub-username/books-recommender:latest

# On EC2 — pull and run without cloning
docker pull your-dockerhub-username/books-recommender:latest
docker run -d -p 8501:8501 your-dockerhub-username/books-recommender:latest
```

### Dockerfile Summary

```dockerfile
FROM python:3.7-slim-buster
EXPOSE 8501
RUN apt-get update && apt-get install -y build-essential software-properties-common git
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 13. Business Applications & Other Domains

### Primary Use Case — Book Discovery

| User | Value Delivered |
|------|----------------|
| **Readers** | Instant personalised suggestions based on books they already love — not generic bestsellers |
| **Booksellers / retailers** | Recommendation widget on product pages increases basket size and session duration |
| **Libraries** | Suggest related titles when a borrower checks out a book — improve collection utilisation |
| **Publishers** | Understand which titles form natural reading clusters to inform marketing and bundling |
| **Reading apps** | In-app "readers who liked X also liked…" feature built on the same KNN engine |

### Adjacent Domains (Same Collaborative Filtering Pattern)

| Domain | Analogous System | Key Adaptation |
|--------|-----------------|---------------|
| **Streaming (Netflix/Spotify)** | Movie or song recommendations | Replace books with content IDs; same pivot + KNN structure |
| **E-commerce** | Product recommendations ("customers also bought") | Replace book ratings with purchase/view events |
| **News / articles** | Content recommendation engine | Use implicit signals (clicks, read-time) as ratings |
| **Academic research** | Paper recommendation in digital libraries | Citation co-occurrence as the rating signal |
| **Restaurants / travel** | "People who liked X also liked Y" | Yelp-style ratings as input matrix |
| **HR / talent** | Job–candidate matching | Skill ratings as the collaborative signal |
| **Healthcare** | Clinical pathway suggestion (with strict governance) | Replace ratings with treatment outcome signals |

---

## 14. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Matrix Factorisation (SVD)** | 🔴 High | SVD (via `surprise` library) typically outperforms basic KNN — decomposes the rating matrix into latent user and item factors |
| **Implicit feedback handling** | 🔴 High | Most users have 0 ratings for most books — use `implicit` library's ALS model designed for sparse implicit data |
| **Hybrid filtering** | 🟡 Medium | Combine collaborative filtering with content-based signals (genre, author, description embeddings) for cold-start resilience |
| **Tune NearestNeighbors** | 🟡 Medium | Experiment with `metric='cosine'` explicitly, and vary `n_neighbors`; add cross-validation |
| **Rating threshold tuning** | 🟡 Medium | The 200-user / 50-book thresholds are arbitrary; tune them via grid search to maximise recommendation diversity and coverage |
| **User-based filtering** | 🟢 Low | Add the alternative approach: find users similar to the current user and recommend what they liked |

### 🏗️ Engineering & MLOps Improvements

| Area | Recommendation |
|------|---------------|
| **Add GitHub Actions CI/CD** | Automate Docker build → ECR push → EC2 deploy on every push to `main` |
| **Experiment tracking** | Integrate MLflow to log KNN hyperparameters and coverage/diversity metrics across runs |
| **Data versioning** | Use DVC to version `books_data.zip` and the serialised artifacts |
| **Online retraining** | Replace the blocking Streamlit training button with a background task (Celery/threading) |
| **Unit tests** | Add `pytest` for pipeline stages, especially data filtering logic |
| **Update data source URL** | Move `books_data.zip` to S3 so `config.yaml` URL works for any developer cloning fresh |
| **Upgrade Python version** | Python 3.7 is EOL; upgrade to 3.10+ and update pickle artifacts accordingly |

### 📦 Product Improvements

- Show **rating counts and average** alongside cover images
- Add a **"Why this recommendation?"** explainer — e.g. "25 readers who loved both books"
- Support **multi-book input** — "I liked these 3 books, what should I read next?"
- Add **genre/author filter** on top of KNN results
- Track **user session clicks** to improve recommendations over time (implicit feedback loop)
- Add **pagination** — show more than 5 recommendations on demand

---

## 15. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `ModuleNotFoundError: books_recommender` | Run `pip install -r requirements.txt` which installs via `-e .` |
| `FileNotFoundError: model.pkl` | Run `python main.py` to execute the full pipeline and generate all artifacts |
| `UnpicklingError` when loading `.pkl` files | Python version mismatch — ensure you are using Python 3.7.10 (same version used to create the pickles) |
| `FileNotFoundError: books_data.zip` | Confirm `books_data.zip` is in the project root, or update `config.yaml` `dataset_download_url` |
| `KeyError` in `recommend_book()` | The selected book is not in `book_pivot.pkl` — retrain the pipeline to rebuild the pivot |
| Streamlit dropdown is empty | `templates/book_names.pkl` missing — run `python main.py` to regenerate |
| `error_bad_lines` deprecation warning | Known issue with pandas >= 1.3 — replace with `on_bad_lines='skip'` in stage 1 |
| Port 8501 already in use | `lsof -ti:8501 \| xargs kill -9` or use `streamlit run app.py --server.port 8502` |
| Docker build fails on `build-essential` | Network issue on EC2 — run `sudo apt-get update` first, then retry |
| EC2 app not accessible | Check Security Group: add inbound rule for TCP port 8501 from `0.0.0.0/0` |
| Book images not loading | `image_url` links in the dataset may be broken (old Book-Crossing URLs) — this is a known data quality issue |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Collaborative Filtering** | Recommendation approach based on collective user behaviour — "users who liked X also liked Y" — without analysing item content |
| **Item-based CF** | Variant of collaborative filtering that measures similarity between items (books) rather than users |
| **User-based CF** | Variant that finds users most similar to the current user and recommends what they liked |
| **Pivot Table** | 2D matrix with book titles as rows, user IDs as columns, and ratings as values — the core data structure for KNN |
| **CSR Matrix** | Compressed Sparse Row matrix — memory-efficient format for storing the sparse pivot table (most entries are 0) |
| **NearestNeighbors** | Scikit-learn class that finds the K most similar items using a distance metric |
| **Cosine Distance** | Measures the angle between two rating vectors — 0 = identical direction, 1 = orthogonal; used by KNN to find similar books |
| **Brute-force KNN** | Exhaustive comparison of every item against every other — preferred for sparse high-dimensional data over tree methods |
| **Sparsity** | The fraction of entries in the pivot table that are zero (unrated) — typically very high (>99%) in real recommender datasets |
| **Cold Start** | Problem when a new user or new item has no ratings — KNN cannot recommend without sufficient rating history |
| **AppConfiguration** | Central class that reads `config.yaml` and assembles typed `namedtuple` configs for each pipeline stage |
| **AppException** | Custom exception class that enriches error messages with the Python filename and line number where the error occurred |
| **namedtuple** | Python immutable record type — used for all config entities (e.g. `DataIngestionConfig`, `ModelTrainerConfig`) |
| **Pickle** | Python object serialisation — saves trained models and data artifacts to `.pkl` files for fast loading at inference |
| **Streamlit** | Python library that turns scripts into interactive web apps — used here for the dropdown, buttons, and image display |
| **Book-Crossing** | Online reading community that collected the BX-Books dataset — source of ratings and book metadata |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---

