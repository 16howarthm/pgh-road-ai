# Pittsburgh Street Closure AI - starter

Two prototypes aligned to the project proposal:

1. **Category-analysis experiment** (`road_ai.py` + notebook): balanced sampling, permit/description de-duplication, stratified prompt-development/test split, zero-/one-/few-shot Jetstream classification, accuracy/macro-F1/confusion matrices, token counts, and model comparison.
2. **Public accountability dashboard** (`dashboard.py`): flags schedules whose recorded `to_date` has passed, maps road segments, provides a permit-level queue, summarizes patterns, and optionally asks an OpenRouter model for a cautious accountability analysis.

## Important data caveats
- The WPRDC data repeats permits across GIS segments, so model evaluation must split **after de-duplication** to avoid leakage.
- The supplied 50-row sample does **not** contain the six proposal labels in a ground-truth category column, so the classification experiment is written to be configurable for the full dataset/actual label column.
- A passed `to_date` is **not proof a road remains physically closed**. The dashboard labels it as a past-due schedule/accountability flag.
- Dates are parsed with `errors='coerce'`; malformed dates (the sample includes `1024-10-20`) are surfaced as data-quality issues rather than silently fixed.

## Run
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENROUTER_API_KEY='your-openrouter-api-key'
export OPENROUTER_MODEL='openai/gpt-oss-120b'   # change to any OpenRouter model you want
.venv/bin/python -m streamlit run dashboard.py
```

The dashboard reads only `pgh data.csv` beside `dashboard.py`. There is no upload, path, or URL selector. Replace that file to refresh the snapshot; source collection time is unknown.

Past due means `to_date` is before today in Pittsburgh and `active` is `t`, `true`, or null (case and surrounding whitespace are normalized). False and unrecognized values are excluded. Missing/invalid end dates are reported separately. The sidebar controls the minimum age, defaulting to 30 days.

The Leaflet map draws each recorded geometry as a separate GeoJSON LineString, with white-outlined blue/orange/red/purple lines and an age legend. Use **Zoom to permit** to inspect a closure; tooltips show `from_street` and `to_street`. The map fits the available geometry automatically. Leaflet loads from unpkg.com and background tiles load from OpenStreetMap without a map API key. Statistics count each identified permit once, using the oldest qualifying end date. All qualifying segments are mapped. Conflicting dates/statuses and missing identifiers are surfaced; this snapshot cannot prove ongoing closures or trends.

The **Category-analysis results** tab provides an executive summary, study process, saved eight-/ten-example OpenRouter results, limitations, and takeaways for city officials. It reads the two reviewed runs in `category_results_fewshot8_openrouter` without making API calls, independently of the queue filter. Results are diagnostic because evaluation errors informed prompt revisions; fresh unseen validation is still required. The evidence panel includes a summary CSV download.

### Analysis configuration
You can put `OPENROUTER_API_KEY=your-openrouter-api-key` in `.env.local` beside `dashboard.py` instead of exporting it. The dashboard loads this file automatically; existing environment variables take precedence. Restart Streamlit after changing it. `.env.local` is ignored by Git.

- `dashboard_data.py`: fixed source path, status/date rules, geometry, permit deduplication.
- `pattern_analysis.py`: age distribution, type/street/applicant/contractor concentrations, data quality, text findings, and actionable proposals. Works without an API key.
- `analysis_backend.py`: OpenRouter model discovery and analysis adapter, independent of Streamlit.
- `prompts/accountability_system.txt`: default system prompt, editable in the dashboard for a session.

Set `OPENROUTER_MODEL` for the initial model. Expand **Model and system prompt** to load the current OpenRouter model catalog, select a model, or enter a custom ID. Click **Generate AI accountability analysis** to send aggregates and top group names, not raw descriptions or the full CSV. API charges may apply. Reports persist across reruns and are hidden when inputs change. API failures leave local analysis available.

The footer links the [WPRDC source](https://data.wprdc.org/dataset/street-closures), [Pittsburgh 311](https://www.pittsburghpa.gov/Resident-Services/311), and [DOMI contact guidance](https://www.pittsburghpa.gov/Business-Development/Mobility-and-Infrastructure/Right-of-Way-Management/Applicant-Guidance).

Run checks with `python -m unittest discover -s tests`.

## Category experiment
Use `category_analysis.ipynb` or import `road_ai.py`. The first notebook run creates and saves a balanced 120-row prompt-development split and 180-row held-out split in `category_splits/`; later runs load those fixed records. Exact normalized descriptions that occur with conflicting labels are excluded and counted in `split_audit.json`. The zero-, one-, and several-example conditions contain exactly 0, 1, and 6 labeled examples total, and every model/condition is scored on all 180 held-out records. Results, summaries, and all confusion matrices are written to `category_results/`.

### Google Colab
Add `jetstream_api_key` in Colab's **Secrets** panel and enable notebook access before running the notebook.

Open or upload `category_analysis.ipynb` in a fresh Colab runtime and run it from top to bottom. The setup cell anonymously clones the public `main` branch into `/content/pgh-road-ai`, installs `pandas`, `scikit-learn`, and `openai`, and loads `jetstream_api_key`. Results are written to `/content/pgh-road-ai/category_results/`; download that directory before ending the runtime if the results need to persist.

## Classifier demo

`classifier_demo.py` is a small standalone Streamlit interface for classifying one street-closure work description. It reuses the winning experiment configuration: Jetstream `llama-4-scout` with the deterministic six-example prompt selected from the saved development split. Jetstream requests use `https://llm.jetstream-cloud.org/api/`.

For local use, add the following to `.streamlit/secrets.toml`:

```toml
jetstream_api_key = "your-jetstream-api-key"
```

Then run:

```bash
streamlit run classifier_demo.py
```

To deploy on Streamlit Community Cloud, create an app from this public GitHub repository, select `classifier_demo.py` as the entry point, add `jetstream_api_key` under the app's **Secrets** settings, and deploy. The classifier demo is separate from the accountability dashboard.

If every classification fails, check the displayed provider error and the Community Cloud app logs. HTTP 502/503 errors indicate a Jetstream service failure; retry after the provider recovers. HTTP 401/403 errors require checking the app's `jetstream_api_key` secret. The public proxy URL above is the documented endpoint for Community Cloud; Jetstream's direct model endpoints require a Jetstream instance or a tunnel and are not a replacement for this URL on Streamlit Cloud. Logs record the exception type and HTTP status without recording descriptions, API keys, or provider response bodies.

## Vercel backup classifier

Live page: https://pgh-road-ai-backup.vercel.app/

Run `.venv/bin/python classifier_demo_vercel.py` from the repository root, then open http://localhost:8000. Local execution loads keys from `.env.local` when python-dotenv is installed, or from the environment. The page defaults to **Jetstream / gpt-oss-120b** and displays the category, confidence score (0–1), and explanation. Choose a provider to use its saved server-side key, then select a model or enter a custom chat model ID. There is no automatic switch to another provider; each test uses your explicit selection.

| Provider | Server environment variable | Example models |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `gpt-4.1-mini`, `gpt-4.1`, `gpt-4o-mini` |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-4-scout`, `deepseek/deepseek-v3.2`, `deepseek/deepseek-v4.1-flash` |
| Jetstream | `JETSTREAM_API_KEY` | `llama-4-scout`, `gpt-oss-120b`, `muse-glimmer` |

The model list changes with the provider. Custom model IDs must be supported by that provider's chat-completions API and accessible to your account. The old `CLASSIFIER_MODEL` environment variable is no longer used: model choice is explicit in the page. The six-example system prompt is frozen in `vercel_classifier/prompt.json`; a test checks it against the evaluated development split. New provider/model combinations have not been accuracy-evaluated.

The Vercel project root is `vercel_classifier/`. Only the static page in `public/` is served as static content. `api/classify.py` runs the Python backend; `api/config.py` returns available model presets and whether each key is configured, never key values. Keep all three keys in the Vercel project's production environment variables, never in browser code. After changing an environment variable, redeploy. To update through Git, commit and push changes to `main`; the connected Vercel project builds the `vercel_classifier/` root directory. For CLI deployments, upload the repository folder structure with `vercel_classifier/` inside it, matching the project root setting.

On October 5, 2026, the live OpenAI default test reached the API but reported exhausted credits / insufficient quota. Earlier Jetstream tests returned HTTP 502 and OpenRouter reported a key spending limit. Configuration status means a key is present, not that credits or service availability have been verified. Provider outages and key/billing errors are displayed without exposing provider bodies. Input length, output tokens, and request duration are bounded, but there is no persistent application rate limiter. API usage may incur charges against the selected server-side key.

### Public briefing tabs on Vercel

The existing Vercel classifier includes **Closure accountability** and **Category-analysis results** tabs. Share `https://pgh-road-ai-backup.vercel.app/#closures` or `https://pgh-road-ai-backup.vercel.app/#research` to open a tab directly. The briefing uses a saved public-data snapshot, computes overdue ages in Pittsburgh time, and makes no paid AI requests. The classifier retains its provider selection and API behavior.

After updating the source CSV or reviewed results, run `.venv/bin/python vercel_classifier/build_briefing.py` and redeploy the existing `vercel_classifier` project. The export includes eligible active/unknown segment rows and six reviewed experiment summaries, without credentials. Collection date remains unknown; deployment does not refresh the source feed.

The **Populating blank work types** tab (`https://pgh-road-ai-backup.vercel.app/#population`) summarizes the completed 250-permit full-taxonomy run, all 21 observed work types, assignments/abstentions, confidence limits, and a reviewed workflow for filling missing work types. It publishes aggregates only, without writing labels back. The category-analysis tab also includes the original notebook's six-condition output (two models × 0/1/6 examples); the transcribed evidence is saved in `vercel_classifier/evidence/original_category_summary.csv`. Rebuild the briefing to refresh the aggregate evidence.

The closure tab reports recorded work types, blank types, permit-wide classification eligibility, and matching saved full-taxonomy predictions with self-reported confidence. It audits all source segments, preserves existing labels, and marks untested records as not assessed. No new model calls are made.
