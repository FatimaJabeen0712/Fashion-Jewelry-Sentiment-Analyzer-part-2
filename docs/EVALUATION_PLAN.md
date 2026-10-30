# Sentiment evaluation plan

**No accuracy figures ship with this project.** The model has not been measured on real customer comments.
Run this plan on your own data and report the results with the limitations below. Any number you did not
measure yourself must be labeled *illustrative*.

## 1. Build a labeled test subset

- **Source:** comments you are permitted to use, from the brands/platforms you will analyze (not the synthetic demo).
- **Size:** at least **300** comments; aim for roughly **100 per class** (positive / neutral / negative).
  Real data is usually dominated by positive comments, so sample *stratified by the model's predicted label*
  (or oversample negatives and neutrals) and record the sampling method.
- **Exclude** spam and duplicates (use the app's *cleaned* export) so you measure sentiment, not cleaning.
- **Keep apart** from anything used to tune lexicon or rules, otherwise scores are optimistic.

## 2. Label

- Labels: `positive`, `neutral`, `negative` (neutral = questions, factual or no opinion).
- Write short guidelines first, e.g. *"Where can I buy this?" = neutral; "pretty but the clasp broke" = pick the
  dominant opinion and mark mixed comments consistently.*
- Prefer **two independent annotators**; report Cohen's kappa and resolve disagreements by discussion. With one
  annotator, say so.
- File format: CSV with columns `text,label` (template downloadable in the app, *Setup & Methodology*).

## 3. Measure

Upload the file in *Setup & Methodology → Evaluate sentiment on your own labeled sample*. The app reports:

- accuracy, per-class precision / recall / F1 with support, **macro precision / recall / F1**
- the confusion matrix (rows = human label, columns = model)
- warnings for fewer than 300 rows or classes with fewer than 30 examples

Report **macro-F1** alongside accuracy: with imbalanced classes, accuracy alone can look good while the
negative class, the one that drives recommendations, performs poorly.

## 4. Report template (fill with measured values only)

| Item | Value |
|---|---|
| Date, app version, `model_version` | |
| Data source(s), date range, sampling method | |
| Labeled rows (per class) | |
| Annotators / Cohen's kappa | |
| Accuracy | |
| Macro precision / recall / F1 | |
| Per-class F1 (pos / neu / neg) | |
| Largest confusion (e.g. negative → neutral) | |
| Error themes found (sarcasm, slang, language, mixed opinion) | |

## 5. Known limitations to state with any result

- Rule/lexicon-based VADER misreads sarcasm, slang, emojis used ironically, and mixed opinions.
- English-tuned. Roman Urdu, Urdu script and code-mixed text are unvalidated.
- Class imbalance makes per-class estimates noisy; small samples have wide confidence intervals.
- Results apply only to the platform, brand and period sampled.

## 6. Optional: aspect and recommendation checks

Sample 50–100 aspect rows from *Product & Aspect Insights* and have a person mark each as correct / wrong
aspect / wrong polarity. Report the share correct per aspect. For recommendations, confirm that the listed
example comments genuinely support the stated action before using it.

## 7. Improving the model

If macro-F1 is too low, review the confusion matrix and error examples, extend the domain lexicon
(`src/sentiment.py`) or aspect keywords (`src/aspect_analysis.py`), then **re-evaluate on a fresh labeled
sample**. Only consider a heavier open-source model if it is free, documented, and runs acceptably on CPU.
