# Fashion & Jewelry Customer Voice Intelligence

A free, local-first Streamlit app that turns customer comments about a jewelry or clothing brand into
evidence-backed product feedback: what customers like and dislike, which products are discussed most,
recurring complaints, feature requests, improvement recommendations and product hypotheses worth testing.

> **What this is not.** Mentions and positive sentiment are proxies for *interest*, not sales. The app never
> labels anything "best-selling". A "feedback snapshot" is not a verdict on a brand. Product ideas are
> hypotheses, not demand forecasts.

## Features

- **Automatic agent:** enter a brand once; it finds YouTube videos, collects their comments, saves a CSV, reloads
  that CSV and analyzes it, from the app (one button) or the command line.
- **Sources:** YouTube video comments (official YouTube Data API v3), CSV upload with column mapping, and a
  clearly labeled **synthetic** demo dataset. Facebook/Instagram and TikTok are *not implemented* (see
  [Connectors](#connectors-and-what-they-need)); use a CSV export for those.
- **Honest 2,000-comment target:** the count is *valid unique comments after cleaning* (non-empty, not spam,
  de-duplicated, passing your language filter). The app shows the full funnel, never pads data, and requires an
  explicit acknowledgement before analyzing a smaller sample. Demo rows never count toward the target.
- **Analysis:** VADER + a fashion/jewelry lexicon for sentiment; clause-level aspect sentiment (price, design,
  material, comfort, fit, quality, durability, color, packaging, delivery, service, value); product mention
  ranking ("most discussed" / "highest positive response"); recurring complaints; feature-request detection.
- **Recommendations:** each improvement links to the comment IDs that support it, with evidence count,
  priority, confidence and limitations.
- **Exports:** raw, cleaned, analyzed, summary, aspect, product and recommendations CSVs (UTF-8 with BOM) plus a
  JSON report. Optional save to your own private Google Cloud Storage bucket.
- **Reuse:** upload a previous export to re-run the analysis without re-collecting.
- **Evaluation tool:** upload your own manually labeled sample to get accuracy, per-class precision/recall/F1,
  macro-F1 and a confusion matrix. See [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md).

## Architecture and data flow

```
Collect                      Clean                    Analyze                       Present
-------                      -----                    -------                       -------
YouTube API ─┐               preprocessing.py         sentiment.py                  app.py (7 tabs)
CSV upload  ─┼─> normalized ─> clean text, drop  ──> aspect_analysis.py ──> AnalysisResult ──> reporting.py
Demo (synth)─┘   schema        empty/spam/dupes/      product_insights.py   (pipeline.py)      CSV / JSON exports
 collectors/     (schemas.py)  language, funnel       recommendations.py                       storage.py (optional GCS)
```

| Path | Role |
|---|---|
| `app.py` | Streamlit UI (Collect Data, Dashboard, Product & Aspect Insights, Recommendations, Data Explorer, CSV Library / Reuse, Setup & Methodology) |
| `config.py` | Settings and secret loading (environment, `.env`, Streamlit secrets) |
| `src/collectors/` | `base.py` interface, `youtube.py`, `csv_import.py`, `demo.py`, and `meta.py` / `tiktok_research.py` (status-only, not implemented) |
| `src/preprocessing.py` | Cleaning, spam heuristics, de-duplication, approximate language filter, funnel counts |
| `src/sentiment.py`, `aspect_analysis.py`, `product_insights.py`, `recommendations.py` | Explainable NLP baseline |
| `src/agent.py`, `run_agent.py` | The automatic agent (find videos → collect → save CSV → reload → analyze) and its command line |
| `src/pipeline.py`, `reporting.py` | Orchestration and export builders |
| `src/security.py`, `storage.py`, `evaluation.py` | Secret scrubbing and export allow-list, optional GCS, sentiment evaluation |
| `tests/` | Offline test suite (network access is blocked in tests) |

## Windows setup (PowerShell)

Use Python 3.11 or newer.

```powershell
cd fashion-jewelry-sentiment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. CSV upload and demo mode work with **no credentials**, no model
download and no GPU; it runs on a modest 8 GB machine.

To enable YouTube collection locally:

```powershell
Copy-Item .env.example .env          # then edit .env and set YOUTUBE_API_KEY=...
```

Run the tests: `python -m pytest -q`

> **Verified environment:** the project and tests were built and run on Python 3.13 with pandas 3.0,
> Streamlit 1.66, scikit-learn 1.9 and Plotly 7.1. Python 3.11/3.12 and older pandas versions are expected to
> work but were not run.

## Automatic agent (collect → CSV → analyze)

**In the app:** *Collect Data → Automatic agent*. Enter the brand name in the sidebar and click
**Run agent: collect → export CSV → analyze**. Optional: a search phrase (default `<brand> review`), extra video
links, how many videos and comments per video to use.

**From the command line** (PowerShell, virtual environment active, `YOUTUBE_API_KEY` set):

```powershell
python run_agent.py --brand "Hiven Jewelry" --niche Jewelry --target 2000
python run_agent.py --brand "Hiven Jewelry" --videos https://youtu.be/XXXXXXXXXXX --no-discover
python run_agent.py --help
```

What it does, in order:

1. Searches YouTube for videos about the brand (official `search.list`) and keeps only those whose title or
   description mention the brand. Videos you list yourself are always included. Turn the filter off with
   *Only videos that mention the brand* / `--no-brand-filter`.
2. Collects comments video by video and **stops as soon as the target of valid unique comments is reached**
   (or the video budget or API quota runs out). Nothing is padded.
3. Saves `raw_comments.csv` to `data/runs/<time>_<brand>/`, **reads that CSV back**, and analyzes it.
4. Writes the analysis CSVs, `report.json` and an `agent_log.json` (videos used, skipped, errors, call counts)
   into the same folder.

If fewer valid comments than the target are found, the CSV is still saved but analysis only runs when you chose
*analyze the smaller sample anyway* / `--allow-small`; results then carry a sample-size warning. Exit codes:
`0` complete, `2` below target or nothing analyzable, `1` error.

Limits to know: it needs a YouTube API key (no key, no agent); it only reaches YouTube (Instagram, Facebook and
TikTok need official access, so use CSV upload for them); search results can include videos that mention the
brand but are not about its products, so check the video list in the result (or `agent_log.json`) before trusting
the findings. To run it regularly, point Windows Task Scheduler at `.venv\Scripts\python.exe run_agent.py ...`.

## CSV format

Only a **comment text** column is required; everything else is optional and mapped in the app (headers are
auto-suggested). Delimiters `, ; | tab` and UTF-8 / Windows-1252 files are accepted (max 50 MB).

| Field | Typical header(s) | Notes |
|---|---|---|
| text (required) | `text`, `comment`, `review`, `body` | the comment |
| date | `date`, `created_at`, `timestamp` | any parseable date |
| source | `source`, `platform` | otherwise the source label you enter |
| product | `product`, `product_name` | helps product ranking |
| likes / replies | `likes`, `like_count`, `replies` | engagement counts |
| rating | `rating`, `stars` | optional |
| post URL / comment ID | `post_url`, `comment_id` | optional |

Example: [docs/example_comments.csv](docs/example_comments.csv) (also downloadable inside the app). Columns that
could identify commenters (names, profile links) are **not** exported even if present in your upload.

## Connectors and what they need

| Source | Status | What you need |
|---|---|---|
| **YouTube** | Implemented (`commentThreads.list`, top-level comments) | Free API key (below). Quota applies; each call costs 1 unit and returns up to 100 comments. |
| **Facebook / Instagram** | **Not implemented** | Meta's Graph API only returns comments for accounts you own or manage, with access tokens/permissions and possibly app review. Export your own comments (e.g. Meta Business Suite) and upload the CSV. |
| **TikTok** | **Not implemented** | The Research API is approval- and eligibility-restricted and is not generally available to commercial apps. Use a CSV you are permitted to export. |
| **Other websites** | **Not supported** | A brand or product URL is never treated as permission to scrape. URLs entered in the sidebar are labels only. |
| **Google Cloud Storage** | Optional | A private bucket and a service account (separate from YouTube; may incur charges). |

Google Cloud does **not** provide one API that extracts comments from Facebook, Instagram, TikTok or arbitrary
sites. The YouTube Data API covers eligible public YouTube video comments only. Channel and playlist links are
rejected (paste individual video links) rather than silently expanded.

### YouTube API key

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and create or select a project.
2. Enable **YouTube Data API v3**: https://console.cloud.google.com/apis/library/youtube.googleapis.com
3. *APIs & Services → Credentials → Create credentials → API key*.
4. Restrict the key: under *API restrictions* choose **YouTube Data API v3** only.
5. Locally put `YOUTUBE_API_KEY=...` in `.env`. On Streamlit Community Cloud add it under *Secrets*.

Never commit the key. The app sends it in a request header, never in a URL, and it is excluded from logs,
error messages and exports. Per Google's current documentation, each call costs 1 unit; comment and video calls share a default 10,000-unit
daily pool, while `search.list` (used by the agent to find videos) has its own default limit of 100 calls per day.
Check your project's quota page, as these limits can change.

### Optional: Google Cloud Storage

Create a **private** bucket and a service account limited to that bucket (e.g. *Storage Object Creator* +
*Viewer*), uncomment `google-cloud-storage` in `requirements.txt`, and set `GCS_BUCKET_NAME` and
`GCS_SERVICE_ACCOUNT_JSON` (see `.streamlit/secrets.toml.example`). Saving only happens when you click
**Save all exports to Cloud Storage**; nothing is made public. Everyone who can open the deployment shares the
same export folder, so keep the app private.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub (see below).
2. At https://share.streamlit.io choose **New app**, select the repo and branch, and set the main file to `app.py`.
3. *Advanced settings*: pick a Python version (3.11+) and paste your secrets (format in
   `.streamlit/secrets.toml.example`).
4. Deploy. The server disk is **ephemeral**: use the download buttons (or Cloud Storage) and re-upload files
   later via *CSV Library / Reuse*.

## Publish to GitHub (without secrets or data)

```powershell
git init
git add .
git status                      # confirm .env, data/ contents and any *.csv are NOT listed
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `.streamlit/secrets.toml`, service-account files, `data/*` and `*.csv`
(except `docs/*.csv`). If a secret is ever committed, revoke it in Google Cloud Console immediately; deleting
the file from a later commit does not remove it from history.

## Interpreting results

- **Net sentiment** = (positive − negative) / total, shown with a 95% margin of error. It describes the
  collected comments only, from the listed sources and dates.
- **Highest positive response** uses the lower bound of a Wilson interval and needs a minimum number of
  mentions, so tiny samples cannot rank first.
- **Recommendations** are keyword-rule matches with traceable evidence. Read the example comments before acting.
- **Opportunities** need at least 3 supporting requests and are labeled hypotheses with a low-cost validation test.
- **Competitor comparison** shows dataset sizes, sources and date ranges; it runs **no** significance test.

## Limitations and responsible use

- Sentiment and aspect rules misread sarcasm, slang, mixed opinions and non-English text. Roman Urdu and other
  mixed-language comments are **not validated**; the language filter is an approximate heuristic.
- The rules were tuned on synthetic demo phrasing, so expect weaker results on real comments until you evaluate
  on a labeled sample of your own data.
- Collect only data you are permitted to use and follow each platform's terms. Only necessary fields are kept
  (no commenter names or profile links). Honor retention limits and deletion requests, and do not re-share
  exports publicly. Analysis results are held in memory for your session; cleaned data is cached for at most
  one hour (max 8 entries) per server process.
- YouTube may return fewer than 2,000 valid comments for a small brand. Add more video links or import
  another source instead of relaxing the target.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No YouTube API key configured" | Set `YOUTUBE_API_KEY` in `.env` (local) or Secrets (cloud), restart the app. |
| "Video search failed" / search quota used | Search has a small separate daily limit. Paste video links yourself (search is then skipped for them), run the agent less often, or wait for the daily reset. |
| "daily quota is exhausted" | Quota resets daily (Pacific time) or request more in Cloud Console. Comments collected so far are kept. |
| "Access denied (HTTP 403)" | Enable YouTube Data API v3 for the project and check the key's API restrictions. |
| "Comments are disabled" / "Video not found" | That video can't be collected; remove it or use a different one. |
| "Below target" | Add more video links or CSV sources. Analysis of a smaller sample needs the acknowledgement tick-box. |
| "Not a YouTube video link" | Paste individual video URLs/IDs. Channel, playlist, Instagram and TikTok links are not supported. |
| CSV import errors | Make sure a comment-text column exists and is mapped; save as UTF-8 CSV. |
| `Activate.ps1` is blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for that PowerShell window. |
| Model download problems | None expected: no model is downloaded (VADER ships inside the package). |
| Cloud Storage "not configured" | Set both `GCS_BUCKET_NAME` and `GCS_SERVICE_ACCOUNT_JSON` and install `google-cloud-storage`. |

## Step 2 - Digital Advertising Budget Optimizer

Simulates ad data for Facebook, Instagram, Google, TikTok and YouTube, computes CTR / CPC / CPA /
conversion rate / ROAS, compares Pandas vs MapReduce vs Spark from 10k to 10M rows, and recommends how
to split the next budget. See [docs/STEP2_AD_BUDGET_OPTIMIZER.md](docs/STEP2_AD_BUDGET_OPTIMIZER.md).

    python run_ads.py recommend --rows 100000 --budget 1000000
    python run_ads.py benchmark

The data is simulated and labeled as such in the app. Spark is optional (`pip install pyspark`, Java 17+).
