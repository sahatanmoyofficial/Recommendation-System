# Assignments and Quizzes: Books Recommender System

## Part 1: Assignments (Practical Implementation)

These assignments are designed to extend your current project and deepen your understanding of Recommendation Systems and MLOps.

### Assignment 1: Hybrid Recommendation Engine
**Objective**: Enhance the current Collaborative Filtering model by adding Content-Based Filtering.
- **Task**: 
    1. Modify `recommendation_config` to include metadata fields like 'Author' and 'Year'.
    2. Create a new transformation pipeline using `TfidfVectorizer` on the 'Book-Author' column.
    3. Update `app.py` to allow users to toggle between "Collaborative Filtering" (existing) and "Content-Based" (new) modes.
- **Deliverable**: A modified `app.py` and updated pipeline code.

### Assignment 2: Containerization & Deployment
**Objective**: Prepare the application for production deployment.
- **Task**:
    1. Review the existing `Dockerfile`.
    2. Create a `docker-compose.yml` file to run the Streamlit app.
    3. (Optional) Deploy the Docker image to a free tier of a cloud provider (e.g., Render, Railway, or AWS EC2).
- **Deliverable**: Working `docker-compose.yml` and a URL to the live app (if deployed).

### Assignment 3: Automated Testing Integration
**Objective**: Ensure code reliability.
- **Task**:
    1. Create a `tests/` directory.
    2. Write unit tests for `TrainingPipeline` using `pytest`.
    3. Add a GitHub Action workflow `.github/workflows/main.yaml` to run tests on every push.
- **Deliverable**: A passing GitHub Actions run.

---

## Part 2: Quizzes (Conceptual Understanding)

### Quiz 1: Recommender Systems
**Q1. Which algorithm is primarily used in this project for finding similar books?**
a) Linear Regression
b) K-Means Clustering
c) K-Nearest Neighbors (KNN)
d) Decision Trees
*Answer: c) K-Nearest Neighbors (KNN)*

**Q2. What is the role of the 'Book Pivot' table created in the transformation stage?**
a) To store book images.
b) To create a user-item interaction matrix (Users vs Books).
c) To list all book authors.
d) To validate data quality.
*Answer: b) To create a user-item interaction matrix.*

### Quiz 2: MLOps & Pipeline
**Q3. Why do we separate the pipeline into stages (Ingestion, Validation, Transformation, Training)?**
a) To make the code look longer.
b) To ensure modularity, reproducibility, and easier debugging.
c) Because Python requires it.
d) To slower down the execution.
*Answer: b) To ensure modularity, reproducibility, and easier debugging.*

**Q4. What is the purpose of `config.yaml` in this project?**
a) To store the trained model.
b) To define static paths, URLs, and parameters outside the code logic.
c) To write the user interface.
d) To store user passwords.
*Answer: b) To define static paths, URLs, and parameters outside the code logic.*

### Quiz 3: Streamlit & Deployment
**Q5. In `app.py`, what function is used to display the book posters?**
a) `st.write()`
b) `st.image()`
c) `st.poster()`
d) `st.show()`
*Answer: b) `st.image()`*
