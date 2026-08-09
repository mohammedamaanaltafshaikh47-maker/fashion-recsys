"""
popularity.py
--------------
Algorithm 1: Popularity-Based Recommender

Kya: Sabse zyada "purchase" hue products ko recommend karta hai.
Kyu: Cold-start problem solve karta hai -- naye user (jiski koi history nahi hai)
     ke liye bhi ye kaam karta hai, kyunki isse kisi specific user ka data
     chahiye hi nahi -- sirf overall trend dekhta hai.
Kaise: interactions.csv mein "purchase" actions ko product-wise count karke,
       sabse zyada count wale products ko top pe rakhte hain.
"""

import csv
import os
from collections import Counter

# Relative path: chahe ye script kahin se bhi (kisi bhi computer pe) chalaya
# jaaye, ye khud dhoondh lega "data" folder ka sahi location -- apni file
# ke position se 2 folders upar jaake "data/" mein.
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


def get_popular_products(top_n=10):
    """
    Returns top_n products sorted by number of purchases (most popular first).
    Har product ke saath purchase_count aur view_count bhi return karta hai,
    taaki UI mein "1.2k people bought this" jaisa dikha sakein.
    """
    interactions = load_interactions()
    products = load_products()

    purchase_counter = Counter()
    view_counter = Counter()

    for row in interactions:
        pid = row["product_id"]
        if row["action"] == "purchase":
            purchase_counter[pid] += 1
        elif row["action"] == "view":
            view_counter[pid] += 1

    # Sabse zyada purchase wale products, sorted descending
    top_products = purchase_counter.most_common(top_n)

    results = []
    for pid, purchase_count in top_products:
        if pid in products:
            product = products[pid].copy()
            product["purchase_count"] = purchase_count
            product["view_count"] = view_counter.get(pid, 0)
            results.append(product)

    return results


if __name__ == "__main__":
    # Quick test -- ye script direct chalake dekh sakte hain ki kaam kar raha hai ya nahi
    top = get_popular_products(top_n=5)
    print(f"Top {len(top)} popular products:\n")
    for i, p in enumerate(top, 1):
        print(f"{i}. {p['productDisplayName']} | Purchases: {p['purchase_count']} | Views: {p['view_count']}")
