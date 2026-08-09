"""
hybrid.py
---------
Algorithm 5: Hybrid Ensemble

Kya: Content-Based + Collaborative + Matrix Factorization -- teeno algorithms
     ke scores ko weighted combine karke ek final, strongest recommendation
     list banata hai.
Kyu: Har algorithm ki alag strength/weakness hai:
     - Content-Based: property-similarity accha, lekin behavior-blind
     - Collaborative: behavior-pattern accha, lekin cold-start mein fail
     - Matrix Factorization: hidden patterns pakadta hai, lekin sparse data
       mein kamzor ho sakta hai
     Hybrid in teeno ki strength combine karta hai, weakness ko cancel-out
     karta hai. Netflix, Amazon jaisi real companies bhi hybrid approach
     use karti hain.
Kaise:
    1. Teeno algorithms se ek product ke liye scores nikalte hain
    2. Har algorithm ke scores ko 0-1 range mein NORMALIZE karte hain
       (kyunki teeno ka scale alag hai -- TF-IDF cosine 0-1 hai, lekin
       SVD scores kuch aur range mein ho sakte hain -- fair comparison
       ke liye normalize zaroori hai)
    3. Weighted formula se combine karte hain:
       final_score = 0.4 * content_score + 0.3 * collaborative_score + 0.3 * matrix_score
    4. Final score ke hisaab se sort karke top-N return karte hain
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from matrix_factorization import MatrixFactorizationRecommender


def normalize_scores(score_dict):
    """
    Scores ko 0-1 range mein laata hai (min-max normalization).
    Kyu zaroori: Alag algorithms ke scores ka scale alag hota hai --
    bina normalize kiye, ek algorithm ka influence doosre se zyada
    ho sakta hai sirf scale ki wajah se, actual quality ki wajah se nahi.
    """
    if not score_dict:
        return {}
    values = list(score_dict.values())
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}


class HybridRecommender:
    def __init__(self,
                 products_path=None,
                 interactions_path=None,
                 weights=(0.4, 0.3, 0.3)):
        """
        weights = (content_weight, collaborative_weight, matrix_weight)
        Content ko sabse zyada weight (0.4) diya hai kyunki ye hamesha
        kaam karta hai (koi cold-start issue nahi), baaki dono ko 0.3-0.3
        equal weight diya hai.
        """
        self.content_rec = ContentBasedRecommender(products_path)
        self.collab_rec = CollaborativeRecommender(interactions_path, products_path)
        self.matrix_rec = MatrixFactorizationRecommender(interactions_path, products_path)
        self.w_content, self.w_collab, self.w_matrix = weights

    def recommend(self, product_id, user_id=None, top_n=5):
        """
        product_id: jis product ko user abhi dekh raha hai (content + collab ke liye)
        user_id: agar available hai, to matrix factorization bhi include hoga
                 (personalized). Agar user_id nahi hai (naya/anonymous user),
                 to sirf content + collaborative use hoga -- graceful fallback.
        """
        # 1. Content-based scores (product-similarity ke basis pe)
        content_results = self.content_rec.recommend(product_id, top_n=20)
        content_scores = {p["id"]: p["similarity_score"] for p in content_results}

        # 2. Collaborative scores (behavior-similarity ke basis pe)
        collab_results = self.collab_rec.recommend(product_id, top_n=20)
        collab_scores = {p["id"]: p["similarity_score"] for p in collab_results}

        # 3. Matrix factorization scores -- ab "product ke similar" logic use karta hai
        #    (recommend_similar_products), taaki candidates content/collaborative
        #    ke saath overlap karein aur teeno signals genuinely combine ho sakein.
        #    Pehle ye "user ke liye best" logic use karta tha, jo bilkul alag
        #    candidate products deta tha -- isliye combine hone par uska asar
        #    na ke barabar tha. Ye humne debug karke fix kiya.
        matrix_results = self.matrix_rec.recommend_similar_products(product_id, top_n=20)
        matrix_scores = {p["id"]: p["predicted_score"] for p in matrix_results}

        # Sabko normalize karo (0-1 range, fair combination ke liye)
        content_norm = normalize_scores(content_scores)
        collab_norm = normalize_scores(collab_scores)
        matrix_norm = normalize_scores(matrix_scores)

        # Un sabhi product_ids ka union lo jo kisi bhi algorithm ne suggest kiya
        all_candidate_ids = set(content_norm) | set(collab_norm) | set(matrix_norm)
        all_candidate_ids.discard(product_id)  # khud ko exclude karo

        final_scores = {}
        for pid in all_candidate_ids:
            c = content_norm.get(pid, 0)
            co = collab_norm.get(pid, 0)
            m = matrix_norm.get(pid, 0)
            final_scores[pid] = (
                self.w_content * c +
                self.w_collab * co +
                self.w_matrix * m
            )

        # Sort by final hybrid score
        sorted_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_matches = sorted_ids[:top_n]

        products_by_id = {p["id"]: p for p in self.content_rec.products}
        results = []
        for pid, score in top_matches:
            if pid in products_by_id:
                product = products_by_id[pid].copy()
                product["hybrid_score"] = round(float(score), 3)
                results.append(product)

        return results


if __name__ == "__main__":
    hybrid = HybridRecommender()

    sample_pid = hybrid.content_rec.product_ids[0]
    sample_user = hybrid.matrix_rec.user_ids[0]
    sample_product = products_by_id = {p["id"]: p for p in hybrid.content_rec.products}[sample_pid]

    print(f"Selected product: {sample_product['productDisplayName']}")
    print(f"For user: #{sample_user}\n")

    recommendations = hybrid.recommend(sample_pid, user_id=sample_user, top_n=5)

    print(f"Top {len(recommendations)} HYBRID recommendations:\n")
    for i, p in enumerate(recommendations, 1):
        print(f"{i}. {p['productDisplayName']} | Hybrid Score: {p['hybrid_score']}")
