"""
api.py
------
Kya: FastAPI server jo saare 5 recommendation algorithms ko web API
     (HTTP endpoints) ke roop mein expose karta hai, taaki frontend UI
     unse data maang sake.
Kyu: Frontend (browser mein chalne wala JavaScript) direct Python functions
     call nahi kar sakta -- usse ek API (URL-based interface) chahiye.
     Ye "bridge" hai backend logic aur frontend UI ke beech.
Kaise: Har algorithm ke liye ek alag endpoint banate hain
       (/api/recommend/popularity, /api/recommend/content, etc.)
       taaki evaluator UI mein switch karke dekh sake har algorithm
       alag kaise perform karta hai (transparency ke liye zaroori).
Kahan run hota hai: `uvicorn api:app --reload` se, phir
       http://localhost:8000 pe accessible.
"""

import sys
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), "algorithms"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional

from popularity import get_popular_products
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from matrix_factorization import MatrixFactorizationRecommender
from hybrid import HybridRecommender

app = FastAPI(title="Fashion Recommendation System API")

# CORS: taaki frontend (jo alag origin/port pe chal sakta hai) API ko call kar sake
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Saare recommenders ek hi baar, server start hote waqt load karte hain
# (baar baar load karna slow hoga -- ye "startup cost" hai, per-request nahi)
print("Loading recommenders... (ye ek baar hi hota hai, server start pe)")
content_rec = ContentBasedRecommender()
collab_rec = CollaborativeRecommender()
matrix_rec = MatrixFactorizationRecommender()
hybrid_rec = HybridRecommender()
print("All recommenders loaded successfully.")


def load_all_products():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "products.csv")
    products = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    return products


ALL_PRODUCTS = load_all_products()


@app.get("/api/products")
def list_products(limit: int = 50):
    """Frontend ke product-picker dropdown/grid ke liye saare products dikhata hai."""
    return ALL_PRODUCTS[:limit]


@app.get("/api/users")
def list_users():
    """Frontend ko available demo user_ids dikhata hai (personalized testing ke liye)."""
    return collab_rec.user_ids


@app.get("/api/recommend/popularity")
def recommend_popularity(top_n: int = 8):
    """Algorithm 1: Trending/popular products (cold-start ke liye)."""
    return get_popular_products(top_n=top_n)


@app.get("/api/recommend/content")
def recommend_content(product_id: str = Query(...), top_n: int = 8):
    """Algorithm 2: Content-based -- product properties ke similar items."""
    if product_id not in content_rec.product_ids:
        raise HTTPException(status_code=404, detail="Product not found")
    return content_rec.recommend(product_id, top_n=top_n)


@app.get("/api/recommend/collaborative")
def recommend_collaborative(product_id: str = Query(...), top_n: int = 8):
    """Algorithm 3: Collaborative -- behaviorally similar items."""
    if product_id not in collab_rec.product_index:
        return []  # cold-start case: is product ke liye interaction data nahi hai
    return collab_rec.recommend(product_id, top_n=top_n)


@app.get("/api/recommend/matrix")
def recommend_matrix(product_id: str = Query(...), top_n: int = 8):
    """Algorithm 4: Matrix Factorization -- latent/hidden pattern based items."""
    if product_id not in matrix_rec.product_index:
        return []
    return matrix_rec.recommend_similar_products(product_id, top_n=top_n)


@app.get("/api/recommend/hybrid")
def recommend_hybrid(product_id: str = Query(...), user_id: Optional[str] = None, top_n: int = 8):
    """Algorithm 5: Hybrid -- teeno ka weighted combination (best result)."""
    if product_id not in content_rec.product_ids:
        raise HTTPException(status_code=404, detail="Product not found")
    return hybrid_rec.recommend(product_id, user_id=user_id, top_n=top_n)


# --- Frontend static files serve karna (same server se poora app chale) ---
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
