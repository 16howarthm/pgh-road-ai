"""Export public permit fields and reviewed experiment evidence for static hosting."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard_data import DATA_PATH, clean, parse_path
from pattern_analysis import analyze
from category_report import RUNS

out = Path(__file__).resolve().parent / 'public' / 'briefing'
d = clean(pd.read_csv(DATA_PATH))
summary, _, _, _ = analyze(d, d[d.past_due_schedule], 1, summary_date := pd.Timestamp.now(tz='America/New_York').date())
fields = ['permit_id', 'permit_type', 'primary_street', 'work_description', 'applicant_name', 'contractor_name', 'from_date', 'to_date', 'active_norm', 'work_type']
# Audit the complete permit, including rows outside the overdue queue.
source_audit = {}
eligible_ids = d.loc[d.eligible_active & d.to_date_parsed.notna(), 'permit_id'].dropna().unique()
for permit_id, group in d[d.permit_id.isin(eligible_ids)].groupby('permit_id'):
    types = sorted(set(group.work_type.dropna().astype(str).str.strip()) - {''})
    descriptions = group.work_description.fillna('').astype(str).str.strip()
    source_audit[str(permit_id)] = {
        'recorded_types': types,
        'classification_eligible': bool(not types and descriptions.ne('').all() and group.work_description.nunique() == 1),
    }
prior = pd.read_csv(ROOT / 'category_population_250/7dfb1a306b9f48e6/permit_predictions.csv')
prior_by_id = {str(r['permit_id']): r for r in prior.to_dict('records')}
rows = []
for record in d[d.eligible_active & d.to_date_parsed.notna()].to_dict('records'):
    row = {key: None if pd.isna(record.get(key)) else str(record[key]) for key in fields}
    row['end_date'] = record['to_date_parsed'].strftime('%Y-%m-%d')
    row['path'] = parse_path(record['geometry'])
    row['from_street'] = str(record.get('from_street') or '')
    row['to_street'] = str(record.get('to_street') or '')
    row['type_audit'] = source_audit.get(row['permit_id'], {'recorded_types': [], 'classification_eligible': False})
    saved = prior_by_id.get(row['permit_id'])
    row['saved_prediction'] = None
    if saved and saved['work_description'] == row['work_description'] and saved['permit_type'] == row['permit_type']:
        confidence = saved.get('confidence_score')
        row['saved_prediction'] = {
            'category': saved['predicted'] if pd.notna(saved['predicted']) else None,
            'confidence': float(confidence) if pd.notna(confidence) else None,
            'status': saved['prediction_status'],
            'reason': saved['reason'] if pd.notna(saved['reason']) else None,
            'run': '7dfb1a306b9f48e6', 'model': 'muse-glimmer',
        }
    rows.append(row)
results = []
for prompt, folder in RUNS.items():
    frame = pd.read_csv(folder / 'classification_summary.csv')
    frame['prompt'] = prompt
    results.extend(frame.to_dict('records'))
payload = {'records': rows, 'quality': summary['data_quality'], 'results': results,
           'exported_on': str(summary_date), 'snapshot_collected_on': None}
(out / 'snapshot.json').write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')))
pd.DataFrame(results).to_csv(out / 'category-summary.csv', index=False)
print(f'Exported {len(rows)} eligible segment rows and {len(results)} experiment summaries.')

# Preserve aggregate evidence only; publish no new permit-level prediction records.
baseline = pd.read_csv(Path(__file__).resolve().parent / 'evidence/original_category_summary.csv')
(out / 'original-category-summary.csv').write_text(baseline.to_csv(index=False))
(out / 'original-results.json').write_text(baseline.to_json(orient='records'))
pop_dir = ROOT / 'category_population_250/7dfb1a306b9f48e6'
settings = json.loads((pop_dir / 'run_settings.json').read_text())
predictions = pd.read_csv(pop_dir / 'permit_predictions.csv')
source = pd.read_csv(DATA_PATH, low_memory=False)
missing = source.work_type.fillna('').str.strip().eq('')
usable = source.work_description.fillna('').str.strip().ne('')
valid_id = source.permit_id.notna() & source.permit_id.astype(str).str.strip().ne('')
audit = source.assign(missing=missing, usable=usable).loc[valid_id].groupby('permit_id').agg(
    all_missing=('missing','all'), all_usable=('usable','all'), descriptions=('work_description','nunique'))
population = {
    'summary': json.loads((pop_dir / 'analysis_summary.json').read_text()),
    'work_types': [c for c in settings['allowed_categories'] if c != 'OUT OF SCOPE'],
    'category_counts': predictions.predicted.value_counts().to_dict(),
    'permit_types': predictions.permit_type.value_counts().to_dict(),
    'additional_examples': len(json.loads((pop_dir / 'additional_examples.json').read_text())),
    'source_rows': len(source), 'missing_rows': int(missing.sum()),
    'unique_permits': len(audit), 'entirely_uncategorized': int(audit.all_missing.sum()),
    'eligible_permits': int((audit.all_missing & audit.all_usable & audit.descriptions.eq(1)).sum()),
    'sample_segment_rows': int(predictions.segment_rows.sum()),
    'thresholds': pd.read_csv(pop_dir / 'confidence_thresholds.csv').to_dict('records'),
    'abstention_causes': pd.read_csv(pop_dir / 'abstention_causes.csv').to_dict('records'),
    'run_id': pop_dir.name,
}
(out / 'population.json').write_text(json.dumps(population, allow_nan=False))
pd.DataFrame(list(population['category_counts'].items()), columns=['Proposed category or abstention','Permits']).to_csv(out / 'population-summary.csv', index=False)
print('Exported original six-condition scores and full-taxonomy population aggregates.')
