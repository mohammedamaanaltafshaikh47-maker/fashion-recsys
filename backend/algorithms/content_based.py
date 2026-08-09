"""
content_based.py
-----------------
Algorithm 2: Content-Based Filtering

Kya: Product ki properties (category, color, usage, articleType, etc.) dekh kar
     us jaise dusre products dhoondhta hai.
Kyu: Personalization ka pehla step -- user jis product mein interest dikhata hai,
     usi jaisi cheezein recommend karne ke liye. Cold-start problem nahi hota
     (kisi bhi single product se turant similar items mil jaate hain).
Kaise:
    1. Har product ki text properties ko ek "combined text" mein jodte hain
       (gender + category + subCategory + articleType + colour + season + usage)
    2. TF-IDF (Term Frequency-Inverse Document Frequency) se in texts ko
       numeric vectors mein convert karte hain -- ye technique batati hai
       ki kaunse words zyada "distinctive/important" hain.
    3. Cosine Similarity nikalte hain -- do products ke vectors ke beech
       ka "angle" measure karta hai. Angle jitna chhota, similarity utni zyada.
    4. Diye gaye product ke sabse zyada similar products return karte hain.
"""

import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def load_products(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "products.csv")
    products = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    return products


def build_combined_text(product):
    """
    Har product ki important text-properties ko ek string mein jodta hai.
    Ye string hi TF-IDF ka 'input document' banega.
    """
    return " ".join([
        product["gender"],
        product["masterCategory"],
        product["subCategory"],
        product["articleType"],
        product["baseColour"],
        product["season"],
        product["usage"],
    ])


class ContentBasedRecommender:
    def __init__(self, products_path=None):
        self.products = load_products(products_path)
        self.product_ids = [p["id"] for p in self.products]
        self.combined_texts = [build_combined_text(p) for p in self.products]

        # TF-IDF vectorizer: text ko numeric vectors mein convert karta hai
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.combined_texts)

        # Sabhi products ke beech pairwise cosine similarity pehle se calculate
        # kar lete hain -- taaki baar baar recommend call karne pe fast rahe.
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, product_id, top_n=5):
        """
        Diye gaye product_id ke liye top_n sabse similar products return karta hai.
        """
        if product_id not in self.product_ids:
            return []

        idx = self.product_ids.index(product_id)

        # Us product ki similarity scores sabhi products ke saath
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))

        # Khud ko exclude karke, similarity ke hisaab se descending sort
        similarity_scores = [s for s in similarity_scores if s[0] != idx]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        top_matches = similarity_scores[:top_n]

        results = []
        for i, score in top_matches:
            product = self.products[i].copy()
            product["similarity_score"] = round(float(score), 3)
            results.append(product)

        return results


if __name__ == "__main__":
    recommender = ContentBasedRecommender()

    # Test: kisi ek product ke liye similar products dhoondte hain
    sample_product_id = recommender.product_ids[0]
    sample_product = recommender.products[0]

    print(f"Selected product: {sample_product['productDisplayName']} "
          f"({sample_product['gender']}, {sample_product['usage']}, {sample_product['baseColour']})\n")

    recommendations = recommender.recommend(sample_product_id, top_n=5)

    print(f"Top {len(recommendations)} similar products:\n")
    for i, p in enumerate(recommendations, 1):
        print(f"{i}. {p['productDisplayName']} | Similarity: {p['similarity_score']}")
