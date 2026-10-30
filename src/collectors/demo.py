"""Synthetic demo dataset. ALWAYS labeled synthetic; never counted as real data.

The rows are generated from templates with a fixed seed so tests are
deterministic. A few duplicates, spam rows and empty rows are injected on
purpose so the preprocessing funnel can be demonstrated.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.schemas import DEMO_METHOD, ensure_schema

DEMO_LABEL = "SYNTHETIC DEMO DATA - not real customer feedback"
DEMO_SOURCE = "demo_synthetic"

JEWELRY = ["ring", "necklace", "bracelet", "earrings", "anklet", "pendant", "choker", "bangle"]
CLOTHING = ["dress", "kurta", "hoodie", "abaya", "jacket", "skirt", "jeans", "scarf"]

POSITIVE = [
    "Love this {p}! The design is so elegant",
    "The {p} looks stunning and the quality is great",
    "Beautiful {p}, fits perfectly and so comfortable",
    "Worth every penny, amazing value for money",
    "Packaging was gorgeous and it arrived on time",
    "Customer service was super helpful and replied fast",
    "The color of the {p} is exactly like the photos",
    "Obsessed with my new {p}",
    "Such a lovely {p}, very sturdy and well made",
]
NEGATIVE = [
    "The {p} tarnished after two weeks, very disappointed",
    "The clasp on the {p} broke on the first day",
    "Way too expensive for this quality",
    "Sizing is off, the {p} runs small and the size chart is wrong",
    "Delivery took over three weeks and nobody replied to my messages",
    "The fabric feels cheap and thin, started pilling after one wash",
    "The {p} color faded and looks different from the pictures",
    "Packaging was damaged and the box was crushed",
    "My skin got itchy and red, I think I am allergic to the metal",
    "Not worth the price, the {p} looks cheap",
]
NEUTRAL = [
    "Where can I buy this {p}?",
    "What is the price of the {p}?",
    "Which size should I order for the {p}?",
    "Is this available in silver?",
    "Saw this {p} on a post today",
]
REQUESTS = [
    "Please make the {p} in more sizes",
    "I wish it came in rose gold and silver",
    "Would be better if you offered a waterproof version",
    "Please add personalized engraving options",
    "Can you please restock the {p}?",
    "Would love matching sets for gifting, please make bundles",
]
SUFFIXES = ["", " \U0001F60D", " !!", " Highly recommend", " Will buy again", " ❤️",
            " Thank you", " Honestly", " for real"]
DETAILS = ["", ". Got it last month", ". Ordered it for my sister", ". Bought it as a gift",
           ". Second order from this shop", ". Saw it on a video first", ". Wore it to a wedding",
           ". Using it daily", ". Took a week to decide", ". My friend recommended the shop",
           ". Ordered online", ". Came with a small card"]
SPAM = [
    "Subscribe to my channel for free followers http://spam.example/x",
    "CLICK THE LINK earn $500 daily http://spam.example/y http://spam.example/z",
    "aaaaaaaaaaaaaaaaaaaaaaaa",
]
EMPTY = ["", "   ", "<br>"]


def generate_demo_dataset(niche: str = "both", n: int = 600, seed: int = 42,
                          brand: str = "Demo Brand",
                          base_date: datetime | None = None) -> pd.DataFrame:
    """Return a normalized synthetic dataset of about ``n`` rows (clearly labeled)."""
    rng = random.Random(seed)
    niche = niche.lower()
    products = JEWELRY if niche == "jewelry" else CLOTHING if niche == "clothing" else JEWELRY + CLOTHING
    base = base_date or datetime(2026, 1, 1, tzinfo=timezone.utc)

    rows: list[dict] = []
    for i in range(n):
        roll = rng.random()
        pool = (POSITIVE if roll < 0.48 else NEGATIVE if roll < 0.74
                else NEUTRAL if roll < 0.86 else REQUESTS)
        product = rng.choice(products)
        text = rng.choice(pool).format(p=product) + rng.choice(DETAILS) + rng.choice(SUFFIXES)
        rows.append({
            "comment_id": f"demo-{i + 1:05d}",
            "text": text,
            "created_at": base - timedelta(days=rng.randint(0, 180), hours=rng.randint(0, 23)),
            "source": DEMO_SOURCE,
            "brand": brand,
            "product_name": "",
            "post_url": f"demo://post-{rng.randint(1, 12)}",
            "like_count": rng.choice([0, 0, 0, 1, 2, 3, 5, 10]),
            "collection_method": DEMO_METHOD,
        })

    # Injected noise so the cleaning funnel has something to show.
    for j, text in enumerate(SPAM + EMPTY):
        rows.append({"comment_id": f"demo-noise-{j}", "text": text, "created_at": base,
                     "source": DEMO_SOURCE, "brand": brand, "collection_method": DEMO_METHOD})
    for j in range(max(1, n // 40)):
        dup = dict(rows[j])
        dup["comment_id"] = f"demo-dup-{j}"
        rows.append(dup)

    return ensure_schema(pd.DataFrame(rows), default_source=DEMO_SOURCE,
                         collection_method=DEMO_METHOD)


def is_demo(df: pd.DataFrame) -> bool:
    """True if any row is synthetic demo data."""
    return "collection_method" in df.columns and bool((df["collection_method"] == DEMO_METHOD).any())
