"""Public-facing synthesis of the saved category experiments; no model requests."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RUNS = {
    "8 examples": ROOT / "category_results_fewshot8_openrouter/82af8ee5b5050713",
    "10 examples": ROOT / "category_results_fewshot8_openrouter/d1829a1157bb0c42",
}


def load_results():
    """Use explicit reviewed runs, rather than selecting a partial run by timestamp."""
    frames = []
    for label, directory in RUNS.items():
        frame = pd.read_csv(directory / "classification_summary.csv")
        frame["Prompt"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def render_category_report():
    st.header("Understanding street-closure work with AI")
    st.caption("Research briefing for city officials · saved experiment results · independent of the accountability queue filter")
    st.subheader("Executive summary")
    st.write(
        "We tested whether AI can turn permit work descriptions into six consistent work categories, "
        "helping staff organize records and identify descriptions that need review. The process compares "
        "model predictions with recorded permit labels, reviews mistakes, and improves the instructions "
        "with labeled examples. This supports a potential staff-assistance pilot; it does not establish "
        "whether a road is closed, whether work is complete, or whether a permit holder is compliant."
    )
    try:
        results = load_results()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning("Saved category results are unavailable. Restore the experiment output files to display the evidence tables.")
        st.caption(str(exc))
        return
    revised = results[results.Prompt.eq("10 examples")]
    best = revised.loc[revised.accuracy.idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best revised accuracy", f"{best.accuracy:.1%}")
    c2.metric("Permits per model", f"{int(best.n_scored):,}")
    c3.metric("Categories evaluated", "6")
    st.write(
        f"The strongest revised result was {best['model']}, matching recorded labels on "
        f"{round(best.accuracy * best.n_scored)} of {int(best.n_scored)} permits. "
        "These are diagnostic results: evaluation mistakes informed the prompt revisions, so a fresh "
        "unseen sample is needed before claiming performance on new City records."
    )

    st.subheader("Analysis")
    st.markdown("**How the study works**")
    st.markdown("""
1. **Prepare the records.** De-duplicate permits and descriptions to avoid counting GIS segments as separate examples. The saved audit excluded 505 records in 28 description groups with conflicting category labels.
2. **Create a balanced sample.** Select 300 records across MACHINERY, DEMOLITION DUMPSTER, CRANE, BARRICADE, MATERIALS, and SCAFFOLD: 50 per category. Save 120 for prompt development and 180 for evaluation, with no overlapping permits or normalized descriptions.
3. **Compare instructions.** The original notebook tests zero, one, and six labeled examples with two models. Its result CSVs are unavailable in this checkout, so original performance and improvement over that baseline are not reported here.
4. **Review and revise.** Follow-up experiments compare three models through OpenRouter with eight and then ten examples. Eight earlier error-review permits stay excluded, leaving the same 172 evaluation permits for both revisions. The ten revised examples come from the development split.
5. **Score every record.** Failed or invalid responses count as incorrect. Save predictions, prompts, examples, confusion matrices, and error reviews so the findings can be checked.
""")
    st.markdown("**Results from the saved OpenRouter runs**")
    table = results.assign(
        Model=results.model,
        **{"Permits scored": results.n_scored.astype(int),
           "Accuracy (%)": (results.accuracy * 100).round(2),
           "Macro-F1": results.macro_f1.round(3),
           "Failed responses": results.failed_predictions.astype(int)},
    )[["Model", "Prompt", "Permits scored", "Accuracy (%)", "Macro-F1", "Failed responses"]]
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Accuracy is the share matching the recorded label. Macro-F1 balances precision and recall for each category and gives all six categories equal weight; 1 is perfect agreement.")
    chart = results.pivot(index="model", columns="Prompt", values="accuracy") * 100
    st.bar_chart(chart, y_label="Accuracy (%)")
    st.write(
        "The ten-example revision raised muse-glimmer from 80.81% to 83.72%, but gpt-oss-120b "
        "fell from 78.49% to 76.74% and llama-4-scout fell from 75.00% to 72.67%. "
        "Scout's failed responses rose from 8 to 19. Adding examples therefore did not help every model; "
        "category accuracy and response reliability both matter. The separate Jetstream eight-example "
        "run failed on all 172 records for each model and is evidence of an unsuccessful run, not a "
        "meaningful measure of classification ability."
    )
    st.markdown("**What the error review found**")
    st.write(
        "Descriptions often mix material storage, active equipment, lifting, disposal, building access, "
        "and site protection. Frequent confusions included MATERIALS versus BARRICADE and CRANE versus "
        "MACHINERY. Revised examples emphasize the main purpose of the permit rather than a single noun. "
        "Some descriptions omit the equipment implied by the recorded label, and some labels appear "
        "inconsistent. Agreement with those labels is a useful benchmark, but does not independently "
        "verify the actual work on site."
    )
    st.info("The balanced research sample does not represent the frequency of work types across the city. These results do not measure staff time saved, cost savings, enforcement outcomes, or performance on a fresh unseen sample.")

    st.subheader("Takeaways")
    st.markdown("""
- **Pilot assisted triage.** Use suggested categories to help staff organize incoming records, with human review before categories affect public reporting or operational decisions.
- **Validate on new records.** Freeze the prompt and evaluate a fresh, independently reviewed sample reflecting the City's real category mix. Report per-category errors and failed responses alongside overall accuracy.
- **Improve the source descriptions.** Encourage explicit statements of the work's main purpose, equipment, storage, disposal, and access needs; resolve conflicting labels with permit staff.
- **Track practical value.** Measure review time, correction rates, response reliability, and API costs in a limited pilot before choosing a model or expanding use.
- **Keep an audit trail.** Preserve the source label, suggested category, model and prompt version, and staff corrections. Use field observations and permit updates to assess closure status separately.
""")
    with st.expander("Evidence and reproducibility"):
        st.write("Source notebooks: category_analysis.ipynb and category_analysis_fewshot8.ipynb. Supporting review: category_analysis_prompt_review.md. Split audit: category_splits/split_audit.json.")
        for label, directory in RUNS.items():
            st.caption(f"{label}: {directory.relative_to(ROOT)}")
        st.write("The tab reads existing saved summaries and makes no AI requests. Partial checkpoint runs are excluded. Queue filters do not change the experiment sample.")
        st.download_button("Download experiment summary (CSV)", table.to_csv(index=False), "category-analysis-summary.csv", "text/csv")
