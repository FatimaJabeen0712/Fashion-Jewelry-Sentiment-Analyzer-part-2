"""Evidence-backed improvement recommendations and future product hypotheses.

Every recommendation carries the comment ids that support it, so it can be
traced back to the Data Explorer. Opportunity ideas are hypotheses inferred
from feedback - not evidence of market demand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from config import DEFAULT_TARGET
from src.product_insights import PRODUCT_TERMS

HYPOTHESIS_LABEL = "Hypothesis inferred from customer feedback; not guaranteed market demand."


@dataclass(frozen=True)
class Rule:
    key: str
    aspect: str
    pattern: str
    title: str
    action: str
    niches: frozenset[str]


_J, _C, _B = frozenset({"jewelry"}), frozenset({"clothing"}), frozenset({"jewelry", "clothing"})

RULES: list[Rule] = [
    Rule("tarnish", "durability", r"tarnish|discolou?r|turned green|rust|fad(?:e|ed|es|ing)|peel",
         "Improve tarnish and finish resistance",
         "Move to thicker PVD / gold-vermeil plating or stainless steel, state plating thickness, "
         "include an anti-tarnish pouch and care card, and back it with a clear warranty.", _J),
    Rule("clasp", "durability", r"clasp|hook|lock|broke|broken|snapped|fell apart|came apart",
         "Strengthen clasps and closures",
         "Switch to stronger lobster / spring-ring clasps, add a safety chain on longer pieces, "
         "and run a pull test on each batch before shipping.", _J),
    Rule("skin_reaction", "comfort", r"allerg|itch|rash|irritat|sensitive|nickel|skin",
         "Offer hypoallergenic, nickel-free materials",
         "Use nickel-free / hypoallergenic metals, label the base metal clearly, and promote a "
         "sensitive-skin range.", _J),
    Rule("fabric", "material", r"fabric|material|thin|see[- ]?through|pilling|rough|cheap|shrunk|shrink|fad",
         "Upgrade fabric quality and describe it accurately",
         "Source heavier or better-finished fabric, pre-shrink garments, and list fabric weight "
         "and composition on each product page.", _C),
    Rule("stitching", "quality", r"stitch|seam|thread|unravel|hem|button|zip",
         "Tighten stitching and finishing quality",
         "Add a seam / stitching inspection step and reinforce stress points before dispatch.", _C),
    Rule("sizing", "fit_sizing", r".",
         "Clarify and widen sizing",
         "Publish a size chart with measurements (ring-size guide or adjustable options for "
         "jewelry; model measurements and more sizes for clothing) and note if items run small or large.", _B),
    Rule("packaging", "packaging", r".",
         "Improve protective, gift-ready packaging",
         "Use rigid boxes or padded mailers, add a pouch for jewelry, and offer a gift option.", _B),
    Rule("delivery", "delivery", r".",
         "Fix delivery speed and communication",
         "Set honest delivery estimates, send tracking promptly, and review courier performance.", _B),
    Rule("service", "customer_service", r".",
         "Strengthen customer service responsiveness",
         "Set a response-time target, document a returns / exchange policy, and follow up on open messages.", _B),
    Rule("color", "color", r".",
         "Make colors match the photos",
         "Photograph in neutral light, add color swatches or alt text for shades, and note screen variation.", _B),
    Rule("price", "price", r".",
         "Review price perception",
         "Check price against comparable items, show what drives the price (materials, craftsmanship), "
         "and consider entry-level options.", _B),
    Rule("value", "value_for_money", r".",
         "Strengthen value for money",
         "Close the gap between price and perceived quality; consider bundles or a warranty.", _B),
]

_GENERIC_ACTION = "Review the example comments and decide on a targeted fix."


def _confidence(n: int) -> str:
    return "high" if n >= 30 else "medium" if n >= 10 else "low"


def _priority(n: int, share: float) -> str:
    if share >= 0.25 or n >= 20:
        return "high"
    if share >= 0.10 or n >= 8:
        return "medium"
    return "low"


def _limitations(n: int, dataset_size: int, target: int) -> str:
    notes = [f"Based on {n} negative mention(s) - keyword rules can miss or misread comments."]
    if dataset_size < target:
        notes.append(f"Dataset has {dataset_size:,} valid comments (target {target:,}); "
                     "treat this as directional only.")
    return " ".join(notes)


def build_recommendations(aspect_rows: pd.DataFrame, niche: str = "both", dataset_size: int = 0,
                          target: int = DEFAULT_TARGET, min_evidence: int = 3,
                          n_examples: int = 3) -> pd.DataFrame:
    """Map negative aspect evidence to improvement actions (traceable to comment ids)."""
    cols = ["recommendation", "action", "aspects", "evidence_count", "share_of_negative",
            "priority", "confidence", "example_comments", "example_comment_ids",
            "evidence_comment_ids", "limitations"]
    if aspect_rows.empty:
        return pd.DataFrame(columns=cols)
    neg = aspect_rows[aspect_rows["aspect_sentiment"] == "negative"]
    if neg.empty:
        return pd.DataFrame(columns=cols)
    niche_set = {"jewelry", "clothing"} if niche.lower() == "both" else {niche.lower()}
    total_neg = len(neg)
    recs = []
    covered: set[str] = set()
    for rule in RULES:
        if not (rule.niches & niche_set):
            continue
        cand = neg[neg["aspect"] == rule.aspect]
        hit = cand[cand["evidence"].str.contains(rule.pattern, case=False, regex=True, na=False)]
        hit = hit.drop_duplicates("comment_id")
        if len(hit) < min_evidence:
            continue
        covered.add(rule.aspect)
        ex = hit.sort_values("aspect_score").drop_duplicates("evidence").head(n_examples)
        share = len(hit) / total_neg
        recs.append({
            "recommendation": rule.title, "action": rule.action, "aspects": rule.aspect,
            "evidence_count": len(hit), "share_of_negative": round(share, 3),
            "priority": _priority(len(hit), share), "confidence": _confidence(len(hit)),
            "example_comments": list(ex["evidence"]), "example_comment_ids": list(ex["comment_id"]),
            "evidence_comment_ids": list(hit["comment_id"]),
            "limitations": _limitations(len(hit), dataset_size, target)})
    # Fallback for complaint aspects with no matching rule (e.g. design_style).
    for aspect, grp in neg.groupby("aspect"):
        if aspect in covered or any(r.aspect == aspect and (r.niches & niche_set) for r in RULES):
            continue
        grp = grp.drop_duplicates("comment_id")
        if len(grp) < min_evidence:
            continue
        ex = grp.sort_values("aspect_score").drop_duplicates("evidence").head(n_examples)
        share = len(grp) / total_neg
        recs.append({
            "recommendation": f"Address {aspect.replace('_', ' ')} complaints",
            "action": _GENERIC_ACTION, "aspects": aspect, "evidence_count": len(grp),
            "share_of_negative": round(share, 3), "priority": _priority(len(grp), share),
            "confidence": _confidence(len(grp)), "example_comments": list(ex["evidence"]),
            "example_comment_ids": list(ex["comment_id"]),
            "evidence_comment_ids": list(grp["comment_id"]),
            "limitations": _limitations(len(grp), dataset_size, target)})
    out = pd.DataFrame(recs, columns=cols)
    order = {"high": 0, "medium": 1, "low": 2}
    return out.sort_values(["priority", "evidence_count"],
                           key=lambda s: s.map(order) if s.name == "priority" else -s
                           ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Future product opportunity hypotheses
# ---------------------------------------------------------------------------
CONCEPTS: dict[str, dict[str, dict[str, str]]] = {
    "waterproof_tarnish_free": {
        "jewelry": dict(concept="Waterproof, tarnish-resistant everyday jewelry line",
                        need="Wear jewelry daily (shower, gym, sweat) without it fading or irritating skin.",
                        features="Stainless steel or thick PVD plating, hypoallergenic, waterproof claim backed by test",
                        differentiator="Verified durability guarantee at an accessible price",
                        risk="Higher unit cost; claim must be provable.",
                        test="Pre-order or waitlist page for 3 designs; track sign-ups and price tolerance."),
        "clothing": dict(concept="Easy-care, wash-resistant everyday basics",
                         need="Garments that keep color and shape after many washes.",
                         features="Pre-shrunk, colorfast fabric, reinforced seams",
                         differentiator="Published wash-test results",
                         risk="Fabric costs may raise price.",
                         test="Small batch of 2 basics; collect 30-wash feedback."),
    },
    "size_range": {
        "jewelry": dict(concept="Adjustable / extended-size range (rings, bracelets, chains)",
                        need="Pieces that fit more people without resizing.",
                        features="Adjustable closures, extender chains, ring-size guide",
                        differentiator="Fit-guaranteed exchange",
                        risk="Adjustable pieces may look less refined.",
                        test="Add 'adjustable' variant to 2 best-discussed items; compare conversion."),
        "clothing": dict(concept="Extended size range (petite to plus)",
                         need="Inclusive sizing with accurate measurements.",
                         features="More sizes, per-size measurements, model sizes",
                         differentiator="Size-inclusive core collection",
                         risk="Inventory complexity.",
                         test="Add 2 extra sizes to one best-discussed style; track returns."),
    },
    "color_options": {
        "jewelry": dict(concept="Multi-metal / multi-color variants of popular pieces",
                        need="Match personal style (gold, silver, rose gold).",
                        features="Same design in several finishes",
                        differentiator="Mix-and-match pricing",
                        risk="Stock splits across variants.",
                        test="Photo-only poll or pre-order for 2 new finishes."),
        "clothing": dict(concept="Wider color palette for best-discussed styles",
                         need="More color choice in styles customers already like.",
                         features="2-3 new seasonal colors",
                         differentiator="Color-accurate photography",
                         risk="Trend risk and dead stock.",
                         test="Limited drop of 2 colors; sell-through after 2 weeks."),
    },
    "personalization": {
        "jewelry": dict(concept="Personalized / engraved jewelry",
                        need="Meaningful gifts with names, initials or dates.",
                        features="Engraving options, fast turnaround, preview tool",
                        differentiator="Made-to-order gift line",
                        risk="Longer lead times; no returns on custom items.",
                        test="Offer engraving on 1 existing product as an add-on."),
        "clothing": dict(concept="Customizable garments (embroidery / initials)",
                         need="Unique, gift-worthy pieces.",
                         features="Embroidery options, preview",
                         differentiator="Small-batch personalization",
                         risk="Production complexity.",
                         test="Embroidery add-on on 1 style."),
    },
    "sets_bundles": {
        "jewelry": dict(concept="Matching sets and gift bundles",
                        need="Coordinated pieces that work together as a gift.",
                        features="Necklace + earrings + bracelet sets, gift box",
                        differentiator="Bundle discount with gift-ready packaging",
                        risk="Lower margin per piece.",
                        test="Create 2 bundles from best-discussed items; compare bundle uptake."),
        "clothing": dict(concept="Co-ord sets and outfit bundles",
                         need="Ready-to-wear coordinated outfits.",
                         features="Matching top/bottom, scarf",
                         differentiator="Complete-look pricing",
                         risk="Size mismatches across pieces.",
                         test="Offer 2 bundles; track attach rate."),
    },
    "gift_packaging": {
        "jewelry": dict(concept="Gift-ready premium packaging option",
                        need="Presentable gifts without extra wrapping.",
                        features="Rigid box, pouch, card message",
                        differentiator="Unboxing experience",
                        risk="Packaging cost.",
                        test="Paid gift-packaging add-on; measure uptake."),
        "clothing": dict(concept="Gift-ready garment packaging",
                         need="Presentable gifts.", features="Gift box, card",
                         differentiator="Unboxing experience", risk="Packaging cost.",
                         test="Paid gift add-on."),
    },
    "new_categories": {
        "jewelry": dict(concept="New audience line (men's / kids' / bridal)",
                        need="Pieces designed for an audience the brand does not yet serve.",
                        features="Category-specific designs and sizing",
                        differentiator="Brand style in a new category",
                        risk="Demand may be niche.",
                        test="Survey followers; pre-order 3 designs."),
        "clothing": dict(concept="New audience line (men's / kids' / bridal)",
                         need="Styles for an audience not yet served.",
                         features="Category-specific sizing", differentiator="Brand style, new audience",
                         risk="Niche demand.", test="Follower poll then small drop."),
    },
}

_ALL_PRODUCT_RX = {n: re.compile("|".join(p for ps in terms.values() for p in ps), re.I)
                   for n, terms in PRODUCT_TERMS.items()}


def _pick_variant(niche: str, texts: list[str]) -> str:
    n = niche.lower()
    if n in ("jewelry", "clothing"):
        return n
    counts = {k: sum(1 for t in texts if rx.search(t)) for k, rx in _ALL_PRODUCT_RX.items()}
    return "clothing" if counts["clothing"] > counts["jewelry"] else "jewelry"


def build_opportunities(requests: pd.DataFrame, niche: str = "both", dataset_size: int = 0,
                        target: int = DEFAULT_TARGET, min_evidence: int = 3) -> pd.DataFrame:
    """Turn grouped feature requests into test-able product hypotheses."""
    cols = ["concept", "theme", "customer_need", "desired_features", "differentiator",
            "risk_assumption", "validation_test", "evidence_count", "example_comments",
            "example_comment_ids", "evidence_comment_ids", "label", "limitations"]
    if requests.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for theme, grp in requests.groupby("theme"):
        if theme not in CONCEPTS or len(grp) < min_evidence:
            continue
        grp = grp.drop_duplicates("comment_id")
        if len(grp) < min_evidence:
            continue
        variant = _pick_variant(niche, list(grp["request_text"]))
        c = CONCEPTS[theme][variant]
        uniq = grp.drop_duplicates("request_text")
        limits = [f"Based on {len(grp)} request(s); requests are not purchase intent."]
        if dataset_size < target:
            limits.append(f"Dataset has {dataset_size:,} valid comments (target {target:,}).")
        rows.append({"concept": c["concept"], "theme": theme, "customer_need": c["need"],
                     "desired_features": c["features"], "differentiator": c["differentiator"],
                     "risk_assumption": c["risk"], "validation_test": c["test"],
                     "evidence_count": len(grp), "example_comments": list(uniq["request_text"].head(3)),
                     "example_comment_ids": list(uniq["comment_id"].head(3)),
                     "evidence_comment_ids": list(grp["comment_id"]),
                     "label": HYPOTHESIS_LABEL, "limitations": " ".join(limits)})
    return pd.DataFrame(rows, columns=cols).sort_values("evidence_count", ascending=False).reset_index(drop=True)
