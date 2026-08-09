"""
metrics.py
----------
Kya: Recommendation system ki quality ko NUMBERS se measure karta hai --
     sirf "kaam karta hai" kehna kaafi nahi, evaluators quantitative proof
     chahte hain (jaisa assignment ke Requirement #5 mein maanga gaya hai).

Metrics jo hum measure karte hain:

1. COVERAGE
   Kya: Poore catalog (300 products) mein se kitne % products KABHI BHI
        kisi recommendation list mein aate hain.
   Kyu: Agar sirf 20% products hi hamesha recommend hote hain, baaki 80%
        "dead stock" ban jaate hain -- kabhi discover hi nahi hote. High
        coverage = healthy, balanced system.

2. DIVERSITY (Intra-list diversity)
   Kya: Ek single recommendation list ke andar kitni "variety" hai
        (sab same articleType, ya mix of categories).
   Kyu: Agar system sirf ek hi type ki cheez baar-baar dikhata hai,
        user bore ho jaata hai -- thodi diversity acchi UX ke liye zaroori.

3. CATEGORY-MATCH PRECISION (Content-Based ke liye)
   Kya: Content-based recommendations mein se kitne % same articleType
        (ya subCategory) match karte hain selected product se.
   Kyu: Ye ek proxy-relevance metric hai -- batata hai recommendations
        "logically related" hain ya random.

Kaise: Har metric ko saare (ya sample) products pe loop karke average
       nikalte hain, taaki ek single overall number mile.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "algorithms"))

from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from matrix_factorization import MatrixFactorizationRecommender
from hybrid import HybridRecommender


def compute_coverage(recommender_fn, sample_product_ids, total_catalog_size, top_n=8):
    """
    recommender_fn: ek function jo product_id leke recommendation list de.
    sample_product_ids: jin products pe hum test chalate hain (query set)
    total_catalog_size: poore catalog ka size (denominator) -- ye sample
        size se ALAG hai, kyunki recommendations sample ke bahar (poore
        300-product catalog) se bhi aa sakte hain.

    Coverage = (kitne UNIQUE products kabhi recommend hue, poore catalog mein se) / (total catalog size)
    """
    recommended_ever = set()
    for pid in sample_product_ids:
        try:
            recs = recommender_fn(pid, top_n)
            for r in recs:
                recommended_ever.add(r["id"])
        except Exception:
            continue
    coverage_pct = (len(recommended_ever) / total_catalog_size) * 100
    return round(coverage_pct, 2), len(recommended_ever)


def compute_diversity(recommender_fn, sample_product_ids, top_n=8):
    """
    Har recommendation list ke andar kitne UNIQUE articleTypes hain,
    list size ke against -- phir sabka average nikalte hain.
    Diversity 1.0 = har item alag category, 0 ke close = sab same category.
    """
    diversity_scores = []
    for pid in sample_product_ids:
        try:
            recs = recommender_fn(pid, top_n)
            if not recs:
                continue
            unique_types = len(set(r["articleType"] for r in recs))
            diversity_scores.append(unique_types / len(recs))
        except Exception:
            continue
    if not diversity_scores:
        return 0.0
    return round(sum(diversity_scores) / len(diversity_scores), 3)


def compute_category_precision(recommender_fn, sample_product_ids, all_products_by_id, top_n=8):
    """
    Content-based ke liye: recommendations mein se kitne % same subCategory
    match karte hain selected product se -- ye proxy-relevance hai.
    """
    precision_scores = []
    for pid in sample_product_ids:
        try:
            source_product = all_products_by_id.get(pid)
            if not source_product:
                continue
            recs = recommender_fn(pid, top_n)
            if not recs:
                continue
            matches = sum(1 for r in recs if r["subCategory"] == source_product["subCategory"])
            precision_scores.append(matches / len(recs))
        except Exception:
            continue
    if not precision_scores:
        return 0.0
    return round(sum(precision_scores) / len(precision_scores) * 100, 2)


def run_evaluation():
    print("=" * 65)
    print("EVALUATION METRICS -- Fashion Recommendation System")
    print("=" * 65)

    content_rec = ContentBasedRecommender()
    collab_rec = CollaborativeRecommender()
    matrix_rec = MatrixFactorizationRecommender()
    hybrid_rec = HybridRecommender()

    all_products_by_id = {p["id"]: p for p in content_rec.products}
    all_product_ids = content_rec.product_ids

    # Speed ke liye, coverage/diversity poore 300 products pe nahi,
    # ek reasonable sample (100) pe compute karte hain
    sample_ids = all_product_ids[:100]

    algorithms = {
        "Content-Based": lambda pid, n: content_rec.recommend(pid, top_n=n),
        "Collaborative": lambda pid, n: collab_rec.recommend(pid, top_n=n),
        "Matrix Factorization": lambda pid, n: matrix_rec.recommend_similar_products(pid, top_n=n),
        "Hybrid": lambda pid, n: hybrid_rec.recommend(pid, user_id=None, top_n=n),
    }

    print(f"\nEvaluating on {len(sample_ids)} sample products (out of {len(all_product_ids)} total)\n")

    print(f"{'Algorithm':<22}{'Coverage %':<14}{'Diversity':<12}{'Category Precision %'}")
    print("-" * 65)

    for name, fn in algorithms.items():
        coverage_pct, unique_count = compute_coverage(fn, sample_ids, len(all_product_ids), top_n=8)
        diversity = compute_diversity(fn, sample_ids, top_n=8)
        precision = compute_category_precision(fn, sample_ids, all_products_by_id, top_n=8)
        print(f"{name:<22}{coverage_pct:<14}{diversity:<12}{precision}")

    print("\n" + "=" * 65)
    print("HOW TO READ THESE NUMBERS")
    print("=" * 65)
    print("""
Coverage %          -- Higher is better. Shows what % of the catalog gets
                        surfaced at least once across recommendations
                        (low coverage = many products never get discovered).

Diversity (0-1)      -- Higher means more variety within a single
                        recommendation list (not just one repeated category).

Category Precision % -- Higher means recommendations are logically related
                        to the source product (relevant, not random).
                        NOTE: For Collaborative/Matrix/Hybrid, lower precision
                        here is EXPECTED and not a bug -- these algorithms
                        intentionally use behavior signals instead of category
                        signals, so they surface cross-category items that
                        content-based filtering would miss.
""")


if __name__ == "__main__":
    run_evaluation()
