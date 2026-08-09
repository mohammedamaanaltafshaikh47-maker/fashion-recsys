# StyleFind — AI-Powered Fashion Recommendation System

A hybrid fashion recommendation engine built with 5 algorithms (Popularity, Content-Based,
Collaborative Filtering, Matrix Factorization, and a Hybrid Ensemble), served via a FastAPI
backend and a Myntra-style browser testing interface.

## Project Structure

```
fashion-recsys/
  data/
    generate_data.py       # synthetic dataset generator (schema based on Kaggle Fashion Product Images)
    products.csv           # 300 fashion products
    interactions.csv       # 540 simulated user view/purchase interactions
  backend/
    algorithms/
      popularity.py         # Algorithm 1: Popularity-Based Recommender
      content_based.py      # Algorithm 2: Content-Based Filtering (TF-IDF + Cosine Similarity)
      collaborative.py      # Algorithm 3: Item-Item Collaborative Filtering
      matrix_factorization.py # Algorithm 4: Matrix Factorization (SVD)
      hybrid.py              # Algorithm 5: Hybrid Ensemble
    tests/
      test_cases.py         # automated successful + failure scenario tests
    evaluation/
      metrics.py             # Coverage, Diversity, Category Precision metrics
    api.py                  # FastAPI server (serves both API + frontend)
    requirements.txt
  frontend/
    index.html              # testing interface (product grid + algorithm tabs)
  docs/
    StyleFind_Documentation.docx
    Bonus_Challenge_Myntra_Comparison.docx
```

## Setup & Run Instructions

**Requirements:** Python 3.9+ installed on your machine.

### 1. Install dependencies

It is strongly recommended to use a virtual environment to avoid conflicts with
other Python packages already installed on your system:

```bash
cd fashion-recsys/backend
python -m venv venv

# Activate it:
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Run the server

```bash
python -m uvicorn api:app --reload
```

### 3. Open the app

Visit **http://localhost:8000** in your browser. The full testing interface
(product catalog, search, category filters, and algorithm tabs) will load automatically —
the same server serves both the API and the frontend.

## Running Tests & Evaluation Separately

```bash
# From inside fashion-recsys/backend, with the virtual environment activated:
python tests/test_cases.py         # runs 9 automated test cases
python evaluation/metrics.py       # prints Coverage / Diversity / Precision metrics
```

## Regenerating the Dataset (optional)

```bash
cd fashion-recsys/data
python generate_data.py
```

This regenerates `products.csv` and `interactions.csv` from scratch (deterministic —
uses a fixed random seed, so output is reproducible).

## How to Use the Testing Interface

1. Browse or search the product catalog on the homepage.
2. Click any product card to select it.
3. The Recommendation Panel appears below, showing results for the currently selected algorithm.
4. Switch between the 5 algorithm tabs (Popularity, Content-Based, Collaborative,
   Matrix Factorization, Hybrid) to compare how each one recommends differently
   for the same product.
5. On the Hybrid tab, use the User dropdown to simulate different users and see
   how personalization changes the results.

## Troubleshooting

- **`uvicorn: command not found`** → use `python -m uvicorn api:app --reload` instead of `uvicorn api:app --reload`.
- **`ModuleNotFoundError`** → make sure you are running the command from inside the `backend/` folder, with all project folders (`algorithms/`, `data/`, etc.) intact in their original relative positions — do not move `api.py` on its own.
- **`numpy.dtype size changed` / binary incompatibility errors** → this means conflicting package versions are installed globally. Use a virtual environment (see Setup step 1) — this isolates the project's dependencies from anything else installed on your machine.

## Author's Notes

Dataset is synthetically generated (see `docs/StyleFind_Documentation.docx`, Section 4 and
Section 9, for the full reasoning and assumptions behind this decision).
