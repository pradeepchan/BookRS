# 📚 Book Recommender System  
### Production-Style Matrix Factorization Pipeline (Truncated SVD)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Model-Matrix%20Factorization-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🚀 Overview

This project implements a scalable collaborative filtering recommender system
using Matrix Factorization (Truncated SVD) on the Goodbooks-10K dataset.

The system learns latent user and item representations from historical ratings
and generates Top-K personalized book recommendations.

---

## 🏗 System Architecture

                +---------------------+
                |   ratings.csv       |
                |   books.csv         |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Data Preprocessing  |
                | - Train/Test Split  |
                | - Pivot Matrix      |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Matrix Factorization|
                | Truncated SVD       |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Predicted Rating    |
                | Reconstruction      |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Ranking Engine      |
                | Top-K Selection     |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Evaluation Metrics  |
                | MAP, nDCG, MRR      |
                +---------------------+

---

# 📂 Dataset

Dataset: Goodbooks-10K  
Source: https://github.com/zygmuntz/goodbooks-10k

Required files: 

```
data/
├── ratings.csv     # User-book ratings
├── books.csv     # Book metadata
```

---

# 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

Mac/Linux:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
- pandas
- numpy
- scikit-learn

---

# ▶ How to Run

**Start Jupyter Notebook**:
```bash
jupyter notebook
```

Open `BookRS.ipynb` and run all cells.

---

## 🧠 Model Details

Matrix Factorization using:

```python
from sklearn.decomposition import TruncatedSVD
```

Matrix Factorization approximates:

R ≈ U × V

Where:
- U → latent user preference vectors
- V → latent item characteristic vectors

Latent dimension: 50 components

---

## 📊 Evaluation Strategy

- 80/20 Train-Test Split
- Relevant item defined as rating ≥ 4.0
- Ranking Metrics:
  - MAP@5
  - nDCG@5
  - MRR@5

Example results:
```
MAP@5  ≈ 0.527  
nDCG@5 ≈ 0.591  
MRR@5  ≈ 0.555  
```

---

# 🧪 Example Usage

Recommend for user:

```python
recommend_for_user(user_id=38897)
```

Result for user ID 38897:
```
User ID: 38897
------------------------------------------------------------
Output: Top-5 recommended books:
1. Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) (Book ID: 18, Predicted Rating: 4.585)
2. A Storm of Swords (A Song of Ice and Fire, #3) (Book ID: 135, Predicted Rating: 3.967)
3. The Fellowship of the Ring (The Lord of the Rings, #1) (Book ID: 19, Predicted Rating: 3.285)
4. The Adventures of Huckleberry Finn (Book ID: 58, Predicted Rating: 2.928)
5. Hamlet (Book ID: 125, Predicted Rating: 2.778)
------------------------------------------------------------
Relevance Vector: [1, 1, 1, 1, 0]
Relevant Books in Test: [1, 131, 4, 135, 903, 138, 14, 18, 19, 20, 278, 6167, 3739,
160, 298, 2102, 58, 191, 194, 204, 479, 225, 740, 503]
```

Generate recommendation for 3 random users:

```python
generate_real_examples(k=5, n_examples=3)
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).