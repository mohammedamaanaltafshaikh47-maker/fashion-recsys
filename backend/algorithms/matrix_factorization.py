"""
matrix_factorization.py
------------------------
Algorithm 4: Matrix Factorization (SVD - Singular Value Decomposition)

Kya: Ek advanced collaborative technique jo user-item matrix ke "hidden/latent
     patterns" dhoondhta hai -- patterns jo humein directly nazar nahi aate
     (jaise ek chhupa hua "style preference" jo formal-wear lovers ko connect
     kare, bina explicitly "formal" label diye).
Kyu: Ye Netflix/Amazon jaisi real companies ki core technique hai (Netflix
     Prize competition isi pe based tha). Item-Item collaborative filtering
     (Step 4) sirf DIRECT common-users dekhta hai, lekin SVD un patterns ko
     bhi pakadta hai jo indirect/hidden hote hain -- isse predictions zyada
     accurate ho sakte hain, especially sparse data mein (jab bahut kam
     interactions available hon).
Kaise:
    1. User-Item matrix lete hain (Step 4 jaisa hi -- rows=users, cols=products)
    2. Is matrix ko SVD se 3 chhote matrices mein todte hain:
       U (user-factors) x Sigma (diagonal weights) x V^T (product-factors)
    3. Sirf top-k "latent factors" rakhte hain (dimensionality reduction) --
       ye hi hidden patterns hain
    4. In factors ko wapas multiply karke poora matrix "reconstruct/predict"
       karte hain -- including wo cells jo pehle khaali the
    5. Kisi user ke liye, un products ko recommend karte hain jinka
       predicted score sabse zyada hai (jo user ne abhi tak dekha nahi)
"""

import csv
import os
import numpy as np
from scipy.sparse.linalg import svds

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def load_interactions(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "interactions.csv")
    interactions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interactions.append(row)
    return interactions


def load_products(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "products.csv")
    products = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products[row["id"]] = row
    return products


class MatrixFactorizationRecommender:
    def __init__(self,
                 interactions_path=None,
                 products_path=None,
                 n_factors=15):
        """
        n_factors: kitne "latent/hidden dimensions" rakhne hain.
        Kam factors = zyada generalized (simple) pattern.
        Zyada factors = zyada detailed lekin overfitting ka risk.
        15 ek reasonable middle-ground hai chhote dataset ke liye.
        """
        self.interactions = load_interactions(interactions_path)
        self.products = load_products(products_path)
        self.n_factors = n_factors

        self.user_ids = sorted(set(row["user_id"] for row in self.interactions), key=int)
        self.product_ids = sorted(set(row["product_id"] for row in self.interactions), key=int)

        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.product_index = {pid: i for i, pid in enumerate(self.product_ids)}

        self._build_matrix_and_factorize()

    def _build_matrix_and_factorize(self):
        n_users = len(self.user_ids)
        n_products = len(self.product_ids)
        matrix = np.zeros((n_users, n_products))

        for row in self.interactions:
            u = self.user_index[row["user_id"]]
            p = self.product_index[row["product_id"]]
            weight = 2 if row["action"] == "purchase" else 1
            matrix[u, p] += weight

        self.original_matrix = matrix

        # SVD ke liye k, matrix ke chhote dimension se kam hona chahiye
        k = min(self.n_factors, min(matrix.shape) - 1)

        # svds sparse/dense dono matrix pe kaam karta hai, k sabse bade
        # singular values/factors return karta hai
        U, sigma, Vt = svds(matrix, k=k)
        sigma_diag = np.diag(sigma)

        # Matrix reconstruct/predict karte hain -- ye hi "predicted preference scores" hain
        self.predicted_matrix = np.dot(np.dot(U, sigma_diag), Vt)

    def recommend_for_user(self, user_id, top_n=5):
        """
        Diye gaye user ke liye top_n products recommend karta hai jinke
        predicted scores sabse zyada hain, IN PRODUCTS MEIN SE jo user ne
        abhi tak interact nahi kiya (naye/unseen products discover karwana).
        """
        if user_id not in self.user_index:
            return []

        u_idx = self.user_index[user_id]
        predicted_scores = self.predicted_matrix[u_idx]

        # User ne jo products already dekhe/khareede hain, unhe exclude karte hain
        already_seen = set()
        for row in self.interactions:
            if row["user_id"] == user_id:
                already_seen.add(row["product_id"])

        scored_products = []
        for p_idx, pid in enumerate(self.product_ids):
            if pid not in already_seen:
                scored_products.append((pid, predicted_scores[p_idx]))

        scored_products.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_products[:top_n]

        results = []
        for pid, score in top_matches:
            if pid in self.products:
                product = self.products[pid].copy()
                product["predicted_score"] = round(float(score), 3)
                results.append(product)

        return results

    def recommend_similar_products(self, product_id, top_n=5):
        """
        Ye method hybrid.py mein use hota hai -- ye "is product ke jaise
        latent-space mein kaun se products close hain" nikalta hai, taaki
        Matrix Factorization ko bhi Content-Based/Collaborative jaisa hi
        "product -> similar products" format mein compare kiya ja sake.

        Kaise: Har product ka "item-factor vector" (Vt ka column) leke,
        cosine similarity nikalte hain -- ye batata hai ki latent/hidden
        dimensions mein kaunse products ek dusre ke sabse "close" hain.
        """
        if product_id not in self.product_index:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        idx = self.product_index[product_id]
        # self.predicted_matrix ke columns hi latent-space mein product
        # representation hain -- unhi ko compare karte hain
        item_vectors = self.predicted_matrix.T  # shape: (n_products, n_users)
        sims = cosine_similarity(item_vectors[idx].reshape(1, -1), item_vectors)[0]

        similarity_scores = [(i, s) for i, s in enumerate(sims) if i != idx]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarity_scores[:top_n]

        results = []
        for i, score in top_matches:
            pid = self.product_ids[i]
            if pid in self.products:
                product = self.products[pid].copy()
                product["predicted_score"] = round(float(score), 3)
                results.append(product)

        return results


if __name__ == "__main__":
    recommender = MatrixFactorizationRecommender()

    sample_user = recommender.user_ids[0]
    print(f"Recommendations for User #{sample_user} (based on hidden/latent patterns):\n")

    recommendations = recommender.recommend_for_user(sample_user, top_n=5)

    for i, p in enumerate(recommendations, 1):
        print(f"{i}. {p['productDisplayName']} | Predicted score: {p['predicted_score']}")
