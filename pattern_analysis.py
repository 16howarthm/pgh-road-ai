"""Deterministic permit statistics and grounded narrative, independent of Streamlit."""
import pandas as pd
from dashboard_data import AGE_LABELS, permit_level


def breakdown(permits, column):
    if column not in permits:
        return pd.DataFrame(columns=["permits", "share_pct", "median_days_past_due"])
    d = permits.assign(group=permits[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown"))
    table = d.groupby("group").agg(permits=("permit_id", "nunique"), median_days_past_due=("days_past_due", "median"))
    table["share_pct"] = (100 * table.permits / max(len(permits), 1)).round(1)
    return table.sort_values(["permits", "median_days_past_due"], ascending=False)


def analyze(data, queue, min_days, as_of):
    p = permit_level(queue)
    groups = {c: breakdown(p, c) for c in ["permit_type", "primary_street", "applicant_name", "contractor_name"]}
    conflicts = data.groupby("permit_id")[["active_norm", "to_date"]].nunique(dropna=False).gt(1).any(axis=1).sum()
    quality = {
        "source_rows": len(data),
        "missing_or_invalid_end_date_rows": int(data.to_date_parsed.isna().sum()),
        "missing_permit_id_rows": int((data.permit_id.isna() | data.permit_id.astype(str).str.strip().eq("")).sum()),
        "conflicting_status_or_end_date_permits": int(conflicts),
        "null_status_queue_permits": int(p.active_norm.eq("null").sum()),
    }
    summary = {
        "as_of": str(as_of), "minimum_days_past_due": min_days,
        "past_due_rule": "End date before as_of; active is t, true, or null. Null is unknown, not confirmed active.",
        "permit_count": len(p), "segment_rows": len(queue),
        "median_days_past_due": float(p.days_past_due.median()) if len(p) else None,
        "p90_days_past_due": float(p.days_past_due.quantile(.9)) if len(p) else None,
        "age_bands": {label: int(p.age_band.eq(label).sum()) for label in AGE_LABELS},
        "groups": {c: t.head(10).reset_index().to_dict("records") for c, t in groups.items()},
        "data_quality": quality,
        "method": "One row per permit using oldest qualifying end date. Group counts are not normalized by total permits issued. A single snapshot cannot establish trends, causes, or physical closures.",
    }
    observations = []
    if len(p):
        observations.append(f"{len(p):,} distinct permits meet the filter across {len(queue):,} segment rows. Median overdue age is {summary['median_days_past_due']:.0f} days; the 90th percentile is {summary['p90_days_past_due']:.0f} days.")
        old = summary["age_bands"]["366+ days"]
        observations.append(f"{old:,} permits ({old / len(p):.1%}) are more than a year past their recorded end date. Verify extensions and completion status before treating these as ongoing obstructions.")
        for column, label in [("permit_type", "permit type"), ("primary_street", "street"), ("applicant_name", "applicant")]:
            t = groups[column].drop(index="Unknown", errors="ignore")
            if len(t):
                row = t.iloc[0]
                observations.append(f"The most frequent {label} is {t.index[0]}: {int(row.permits):,} permits ({row.share_pct:.1f}% of this queue). Counts reflect volume, not a violation rate.")
    else:
        observations.append("No identifiable permits meet the current filter. This does not establish that all roads are open.")
    proposals = [
        {"Proposal": "Verify the oldest records", "Evidence": f"{summary['age_bands']['366+ days']} permits exceed one year", "Action / owner": "DOMI: check extensions and completion; publish a verified end date and status.", "Measure": "Share of flagged permits verified; median verification time"},
        {"Proposal": "Improve permit data completeness", "Evidence": f"{quality['missing_or_invalid_end_date_rows']} rows lack a usable end date; {quality['conflicting_status_or_end_date_permits']} permits have conflicting status or dates", "Action / owner": "Data publisher and DOMI: validate dates, reconcile segments, and distinguish unknown from active.", "Measure": "Missing-date rate and conflicting-permit count"},
        {"Proposal": "Coordinate corridor reviews", "Evidence": "Street concentrations in the filtered queue", "Action / owner": "DOMI: review high-count streets for overlapping work and pedestrian access; verify conditions before scheduling changes.", "Measure": "Verified overlapping closures and time to restore access"},
        {"Proposal": "Close the public reporting loop", "Evidence": "This snapshot cannot confirm physical conditions", "Action / owner": "311 and DOMI: link reports to permit IDs and publish resolution updates.", "Measure": "Report acknowledgement and resolution time"},
    ]
    return summary, groups, observations, proposals
