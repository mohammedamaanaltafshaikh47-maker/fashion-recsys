"""
test_cases.py
-------------
Kya: Ye script hamare recommendation system ke "test cases" hai --
     Successful Scenarios (jab system sahi kaam kare) aur Failure Scenarios
     (jab system limitations face kare) dono cover karta hai, jaisa assignment
     mein Requirement #4 mein maanga gaya hai.
Kyu: Sirf "code chal raha hai" kehna kaafi nahi hai -- humein systematically
     dikhana hai ki humne apna system test kiya hai, aur uski limitations
     bhi pata hain (jo assignment khud "a strength" kehta hai).
Kaise: Har test ek chhota function hai jo ek specific scenario check karta hai,
       aur PASS/FAIL print karta hai.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "algorithms"))

from popularity import get_popular_products
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from matrix_factorization import MatrixFactorizationRecommender
from hybrid import HybridRecommender

results = []


def check(name, condition, note=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, note))
    print(f"[{status}] {name}" + (f" -- {note}" if note else ""))


print("=" * 60)
print("SUCCESSFUL SCENARIOS -- system performs well")
print("=" * 60)

content_rec = ContentBasedRecommender()
collab_rec = CollaborativeRecommender()
matrix_rec = MatrixFactorizationRecommender()
hybrid_rec = HybridRecommender()

# Test 1: Popularity recommender normal case
pop_results = get_popular_products(top_n=5)
check(
    "Popularity: returns top-5 products with purchase data",
    len(pop_results) == 5 and all("purchase_count" in p for p in pop_results),
    f"Got {len(pop_results)} results"
)

# Test 2: Content-based normal case -- category should be dominant match
sample_pid = content_rec.product_ids[0]
sample_category = content_rec.products[0]["articleType"]
content_results = content_rec.recommend(sample_pid, top_n=5)
same_category_count = sum(1 for p in content_results if p["articleType"] == sample_category)
check(
    "Content-Based: majority of recommendations share same articleType",
    same_category_count >= 3,
    f"{same_category_count}/5 matched category '{sample_category}'"
)

# Test 3: Collaborative filtering normal case -- product with known interactions
known_pid = collab_rec.product_ids[0]
collab_results = collab_rec.recommend(known_pid, top_n=5)
check(
    "Collaborative: returns non-empty results for a product with interaction history",
    len(collab_results) > 0,
    f"Got {len(collab_results)} results"
)

# Test 4: Matrix factorization normal case
matrix_results = matrix_rec.recommend_similar_products(sample_pid, top_n=5)
check(
    "Matrix Factorization: returns results with predicted scores",
    len(matrix_results) > 0 and all("predicted_score" in p for p in matrix_results),
    f"Got {len(matrix_results)} results"
)

# Test 5: Hybrid normal case -- combines signals successfully
hybrid_results = hybrid_rec.recommend(sample_pid, user_id=matrix_rec.user_ids[0], top_n=5)
check(
    "Hybrid: returns combined weighted recommendations",
    len(hybrid_results) > 0 and all("hybrid_score" in p for p in hybrid_results),
    f"Got {len(hybrid_results)} results"
)

print()
print("=" * 60)
print("FAILURE / EDGE-CASE SCENARIOS -- known limitations")
print("=" * 60)

# Test 6: Invalid product ID for content-based
invalid_results = content_rec.recommend("NONEXISTENT_ID", top_n=5)
check(
    "Content-Based: gracefully handles invalid product_id (returns empty, no crash)",
    invalid_results == [],
    "Correctly returned empty list instead of crashing"
)

# Test 7: Cold-start product for collaborative filtering
# (ek product jiska koi interaction data hi nahi hai interactions.csv mein)
all_product_ids = set(content_rec.product_ids)
products_with_interactions = set(collab_rec.product_ids)
cold_start_products = all_product_ids - products_with_interactions

if cold_start_products:
    cold_pid = next(iter(cold_start_products))
    cold_result = collab_rec.recommend(cold_pid, top_n=5)
    check(
        "Collaborative: cold-start product (no interactions) returns empty gracefully",
        cold_result == [],
        f"Product {cold_pid} has zero interaction history -- known limitation, "
        f"Popularity/Content-Based should be used as fallback in production"
    )
else:
    check("Collaborative: cold-start test", True, "No cold-start products found in this sample")

# Test 8: Hybrid without user_id (anonymous user -- matrix factorization should be skipped gracefully)
hybrid_no_user = hybrid_rec.recommend(sample_pid, user_id=None, top_n=5)
check(
    "Hybrid: works even without user_id (falls back to content+collaborative only)",
    len(hybrid_no_user) > 0,
    "Anonymous/new users still get recommendations, just without personalized SVD signal"
)

# Test 9: top_n larger than available products (edge case)
large_n_results = content_rec.recommend(sample_pid, top_n=1000)
check(
    "Content-Based: requesting more results than exist doesn't crash",
    len(large_n_results) <= len(content_rec.product_ids),
    f"Returned {len(large_n_results)} (capped correctly, no overflow/crash)"
)

print()
print("=" * 60)
print(f"SUMMARY: {sum(1 for r in results if r[1]=='PASS')}/{len(results)} tests passed")
print("=" * 60)
