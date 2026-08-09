"""
collaborative.py
-----------------
Algorithm 3: Item-Item Collaborative Filtering

Kya: Product properties ko IGNORE karta hai. Sirf users ka behavior dekhta hai --
     "jo log Product A ke saath interact karte hain, wo Product B ke saath bhi
     interact karte hain" -- to A aur B ko "similar" maan lete hain.
Kyu: Content-based filtering sirf property-similarity dekh sakta hai (color,
     category). Lekin real users ka taste behavior-driven bhi hota hai --
     jaise "Jeans + Belt" saath khareedna, jo category-wise related nahi
     lagte par behaviorally connected hain. Ye sirf user-interaction data
     se hi pata chal sakta hai.
Kaise:
    1. interactions.csv se ek User-Item matrix banate hain
       (rows = users, columns = products, value = interaction strength)
    2. View = weight 1, Purchase = weight 2 (purchase zyada strong signal hai)
    3. Matrix ko transpose karke item-item similarity nikalte hain
       (cosine similarity -- kitne common users dono products se interact karte hain)
    4. Diye gaye product ke top-N "behaviorally similar" products return karte hain
"""

import csv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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


class CollaborativeRecommender:
    def __init__(self,
                 interactions_path=None,
                 products_path=None):
        self.interactions = load_interactions(interactions_path)
        self.products = load_products(products_path)

        # Unique users aur products ki list (matrix ke rows/columns banane ke liye)
        self.user_ids = sorted(set(row["user_id"] for row in self.interactions), key=int)
        self.product_ids = sorted(set(row["product_id"] for row in self.interactions), key=int)

        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.product_index = {pid: i for i, pid in enumerate(self.product_ids)}

        self._build_matrix()

    def _build_matrix(self):
        """
        User-Item matrix banata hai. View = 1 weight, Purchase = 2 weight
        (kyunki purchase ek zyada strong 'ye product pasand aaya' signal hai
        view ke muqable).
        """
        n_users = len(self.user_ids)
        n_products = len(self.product_ids)
        matrix = np.zeros((n_users, n_products))

        for row in self.interactions:
            u = self.user_index[row["user_id"]]
            p = self.product_index[row["product_id"]]
            weight = 2 if row["action"] == "purchase" else 1
            matrix[u, p] += weight

        self.user_item_matrix = matrix

        # Item-Item similarity: matrix ko transpose karke (products x users banake)
        # cosine similarity nikalte hain -- ab hum products compare kar rahe hain
        item_matrix = matrix.T  # shape: (n_products, n_users)
        self.item_similarity = cosine_similarity(item_matrix)

    def recommend(self, product_id, top_n=5):
        """
        Diye gaye product_id ke liye behaviorally similar top_n products.
        """
        if product_id not in self.product_index:
            return []

        idx = self.product_index[product_id]
        similarity_scores = list(enumerate(self.item_similarity[idx]))

        # Khud ko exclude karke, sort by similarity descending
        similarity_scores = [s for s in similarity_scores if s[0] != idx]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Sirf non-zero similarity wale products (jinke koi common user hi na ho,
        # unhe recommend karne ka koi matlab nahi)
        similarity_scores = [s for s in similarity_scores if s[1] > 0]

        top_matches = similarity_scores[:top_n]

        results = []
        for i, score in top_matches:
            pid = self.product_ids[i]
            if pid in self.products:
                product = self.products[pid].copy()
                product["similarity_score"] = round(float(score), 3)
                results.append(product)

        return results


if __name__ == "__main__":
    recommender = CollaborativeRecommender()

    # Test: pehle product ke liye collaborative recommendations dekhte hain
    sample_pid = recommender.product_ids[0]
    sample_product = recommender.products[sample_pid]

    print(f"Selected product: {sample_product['productDisplayName']}\n")

    recommendations = recommender.recommend(sample_pid, top_n=5)

    if not recommendations:
        print("Koi collaborative match nahi mila (is product ke saath koi common user interaction nahi) -- "
              "isliye ye cold-start jaisa case hai, yahan popularity/content-based fallback zaroori hai.")
    else:
        print(f"Top {len(recommendations)} collaboratively similar products:\n")
        for i, p in enumerate(recommendations, 1):
            print(f"{i}. {p['productDisplayName']} | Similarity: {p['similarity_score']}")
