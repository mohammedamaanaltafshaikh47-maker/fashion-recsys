"""
generate_data.py
-----------------
Kya: Ye script fashion e-commerce ka synthetic (fake but realistic) dataset banata hai,
     jiska SCHEMA (columns) Kaggle ke widely-used "Fashion Product Images" dataset se
     match karta hai: id, gender, masterCategory, subCategory, articleType, baseColour,
     season, year, usage, productDisplayName.
Kyu: Actual Kaggle data hum directly nahi le sakte (bahut bada, aur is environment mein
     Kaggle access nahi hai), isliye hum wahi real-world schema follow karte hue apna
     controlled sample dataset generate karte hain. Docs mein clearly likhenge ki
     structure Kaggle "Fashion Product Images Dataset" (paramaggarwal) se inspired hai.
Kaise: random but realistic combinations se products banaye jaate hain,
       aur users ke fake "view/purchase" interactions generate kiye jaate hain
       (collaborative filtering / matrix factorization ke liye zaroori).
"""

import csv
import random

random.seed(42)  # taaki har baar same data bane (reproducibility)

genders = ["Men", "Women", "Unisex"]
master_categories = ["Apparel", "Footwear", "Accessories"]

sub_category_map = {
    "Apparel": ["Topwear", "Bottomwear", "Dress", "Innerwear"],
    "Footwear": ["Shoes", "Sandals", "Flip Flops"],
    "Accessories": ["Bags", "Belts", "Watches", "Jewellery"],
}

article_type_map = {
    "Topwear": ["Shirts", "Tshirts", "Kurtas", "Sweaters"],
    "Bottomwear": ["Jeans", "Trousers", "Shorts"],
    "Dress": ["Dresses", "Sarees", "Jumpsuits"],
    "Innerwear": ["Vests", "Briefs"],
    "Shoes": ["Casual Shoes", "Sports Shoes", "Formal Shoes"],
    "Sandals": ["Sandals"],
    "Flip Flops": ["Flip Flops"],
    "Bags": ["Handbags", "Backpacks"],
    "Belts": ["Belts"],
    "Watches": ["Watches"],
    "Jewellery": ["Necklaces", "Earrings"],
}

base_colours = ["Navy Blue", "Black", "White", "Blue", "Red", "Beige", "Green", "Pink", "Grey", "Maroon", "Yellow"]
seasons = ["Summer", "Winter", "Fall", "Spring"]
years = [2018, 2019, 2020, 2021, 2022, 2023]
usages = ["Casual", "Formal", "Party", "Sports", "Ethnic"]

brands = ["Peter England", "Turtle Check", "Roadster", "HRX", "Puma", "Van Heusen", "W", "Fabindia"]

products = []
product_id = 10000

for _ in range(300):
    gender = random.choice(genders)
    master_category = random.choice(master_categories)
    sub_category = random.choice(sub_category_map[master_category])
    article_type = random.choice(article_type_map[sub_category])
    base_colour = random.choice(base_colours)
    season = random.choice(seasons)
    year = random.choice(years)
    usage = random.choice(usages)
    brand = random.choice(brands)
    # Price ko round numbers mein rakhte hain (jaise 1500, 2500, 3000) --
    # real e-commerce sites hamesha clean pricing use karte hain,
    # random numbers (jaise 3755, 2104) unprofessional lagte hain.
    price = random.randrange(500, 5001, 100)

    product_display_name = f"{brand} {gender} {usage} {base_colour} {article_type}"

    products.append({
        "id": product_id,
        "gender": gender,
        "masterCategory": master_category,
        "subCategory": sub_category,
        "articleType": article_type,
        "baseColour": base_colour,
        "season": season,
        "year": year,
        "usage": usage,
        "productDisplayName": product_display_name,
        "price": price,
    })
    product_id += 1

with open("/home/claude/fashion-recsys/data/products.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=products[0].keys())
    writer.writeheader()
    writer.writerows(products)

# --- Users ka interaction data (collaborative filtering ke liye) ---
# Har user kuch products "view" ya "purchase" karta hai.
# Hum jaan-bujh kar ek pattern rakhte hain: jo users same "usage" (Casual/Formal/Party/etc.)
# pasand karte hain, unke interactions overlap karte hain -- taaki collaborative filtering
# ko seekhne ke liye kuch signal mile (random data se koi pattern nahi milta).

num_users = 60
interactions = []

# Har user ek "preferred usage" rakhta hai (jaise real users ka taste hota hai)
user_preferred_usage = {u: random.choice(usages) for u in range(1, num_users + 1)}

for user_id in range(1, num_users + 1):
    preferred_usage = user_preferred_usage[user_id]
    # Us usage-type ke matching products dhoondo
    matching_products = [p for p in products if p["usage"] == preferred_usage]
    # Kabhi kabhi user thoda "explore" bhi karta hai (dusre usage-type ke products)
    other_products = [p for p in products if p["usage"] != preferred_usage]

    chosen = random.sample(matching_products, min(6, len(matching_products))) + \
             random.sample(other_products, 3)

    for p in chosen:
        action = random.choice(["view", "view", "purchase"])  # view zyada common, purchase kam
        interactions.append({
            "user_id": user_id,
            "product_id": p["id"],
            "action": action
        })

with open("/home/claude/fashion-recsys/data/interactions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["user_id", "product_id", "action"])
    writer.writeheader()
    writer.writerows(interactions)

print(f"Generated {len(products)} products and {len(interactions)} interactions.")
