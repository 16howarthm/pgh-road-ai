# Pittsburgh Street Closure AI - starter

Two prototypes aligned to the project proposal:

1. **Category-analysis experiment** (`road_ai.py` + notebook): balanced sampling, permit/description de-duplication, stratified prompt-development/test split, zero-/one-/few-shot OpenRouter classification, accuracy/macro-F1/confusion matrices, token counts, and model comparison.
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

The Leaflet map draws each recorded geometry as a separate GeoJSON LineString, with white-outlined blue/orange/red/purple lines and an age legend. Use **Zoom to permit** to inspect a closure; tooltips show `from_street` and `to_street`. The map fits the available geometry automatically. Leaflet loads from unpkg.com and background tiles load from CARTO. Statistics count each identified permit once, using the oldest qualifying end date. All qualifying segments are mapped. Conflicting dates/statuses and missing identifiers are surfaced; this snapshot cannot prove ongoing closures or trends.

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
Because the repository is private, add both of these values in Colab's **Secrets** panel and enable notebook access for each one before running the notebook:

- `GITHUB_TOKEN`: a fine-grained GitHub personal access token limited to `16howarthm/pgh-road-ai`, with read-only repository Contents access.
- `OPENROUTER_API_KEY`: the OpenRouter API key used for model calls.

Open or upload `category_analysis.ipynb` in Colab and run it from top to bottom. The setup cell securely supplies `GITHUB_TOKEN` to Git only for the private clone request; it does not print the token, put it in the clone URL, or save it in the repository's Git configuration. The cell then changes to the repository root and installs only the three experiment dependencies (`pandas`, `scikit-learn`, and `openai`). Outputs are written to `/content/pgh-road-ai/category_results/`; download that directory before ending the runtime if the results need to persist.
