import hashlib
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis_backend import DEFAULT_MODEL, PROMPT_PATH, generate_analysis, list_models
from dashboard_data import AGE_COLORS, AGE_LABELS, DATA_PATH, clean, parse_path, permit_level
from pattern_analysis import analyze
from dashboard_map import map_html

st.set_page_config(page_title="Pittsburgh Road Closure Accountability", layout="wide")
st.title("Pittsburgh Road Closure Accountability")
st.caption("Past-due permits have a recorded end date before today and active status t, true, or null. Null means unknown. A past-due date is not proof that a road remains blocked.")


@st.cache_data(ttl=3600)
def load_data(modified_ns):
    return pd.read_csv(DATA_PATH)


@st.cache_data(ttl=3600)
def available_models():
    return list_models()


as_of = datetime.now(ZoneInfo("America/New_York")).date()
try:
    modified_ns = DATA_PATH.stat().st_mtime_ns
    d = clean(load_data(modified_ns), today=as_of)
except (OSError, ValueError) as exc:
    st.error(f"Could not load pgh data.csv: {exc}")
    st.stop()

st.sidebar.header("Contents")
st.sidebar.markdown("""
- [At a glance](#overview)
- [Map of past-due schedules](#closure-map)
- [Accountability queue](#accountability-queue)
- [Pattern analysis](#pattern-analysis)
- [Proposals for improvement](#proposals)
- [AI pattern analysis and proposals](#ai-analysis)
- [Sources and getting an issue fixed](#sources-and-contact)
""")
st.sidebar.divider()

st.sidebar.header("Queue filter")
min_days = st.sidebar.slider("Minimum days past due", 1, 3650, 30)
q = d[d.past_due_schedule & d.days_past_due.ge(min_days)].copy()
perm = permit_level(q)
summary, groups, observations, proposals = analyze(d, q, min_days, as_of)

map_df = q.copy()
map_df["path"] = map_df.geometry.apply(parse_path)
map_df = map_df[map_df.path.notna()].copy()
map_df["color"] = [AGE_COLORS[AGE_LABELS.index(str(band))] for band in map_df.age_band]

with st.container(border=True):
    st.subheader("At a glance", anchor="overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Past-due permits", f"{len(perm):,}")
    c2.metric("Mapped segments", f"{len(map_df):,}")
    c3.metric("Median days past due", f"{perm.days_past_due.median():.0f}" if len(perm) else "—")
    c4.metric("Max days past due", f"{perm.days_past_due.max():,.0f}" if len(perm) else "—")
    st.caption(f"As of {as_of} (Pittsburgh time). Metrics and analysis follow the minimum-age filter.")

st.write("")

with st.container(border=True):
    st.subheader("Map of past-due schedules", anchor="closure-map")
    if len(map_df):
        focus = st.selectbox("Zoom to permit", ["All qualifying closures"] + sorted(map_df.permit_id.dropna().unique().tolist()))
        components.html(map_html(map_df, None if focus == "All qualifying closures" else focus), height=560)
        st.caption("Lines follow the recorded GIS geometry. Street names describe the reported endpoints; they are not geocoded or used to invent missing road sections.")
    else:
        st.info("No mappable past-due records under the current filter.")
    legend = " &nbsp; ".join(f'<span style="color:rgb({r},{g},{b})">━</span> {label}' for label, (r, g, b) in zip(AGE_LABELS, AGE_COLORS))
    st.markdown(legend, unsafe_allow_html=True)
    st.caption(f"Color indicates days past the recorded end date. {len(q) - len(map_df):,} qualifying segment rows could not be mapped.")

st.write("")

with st.container(border=True):
    st.subheader("Accountability queue", anchor="accountability-queue")
    show = [c for c in ["permit_id", "permit_type", "primary_street", "work_description", "applicant_name", "contractor_name", "from_date", "to_date", "days_past_due", "active"] if c in perm]
    st.dataframe(perm[show], use_container_width=True, hide_index=True)
    st.caption("One row per permit, using its oldest qualifying recorded end date. Missing permit IDs are excluded from permit counts. All qualifying GIS segments remain on the map.")

st.write("")

with st.container(border=True):
    st.subheader("Pattern analysis", anchor="pattern-analysis")
    for observation in observations:
        st.write(observation)
    if len(perm):
        st.bar_chart(pd.Series(summary["age_bands"], name="Permits"))
        for tab, column in zip(st.tabs(["Permit types", "Streets", "Applicants", "Contractors"]), groups):
            with tab:
                st.dataframe(groups[column].head(20), use_container_width=True)
                st.caption("Top 20 groups by distinct permits; share is of the filtered queue. Each permit is assigned the values on its oldest qualifying row.")
    with st.expander("Data quality and interpretation"):
        st.json(summary["data_quality"])
        st.write(summary["method"])
        st.write("Null status is included as requested, but is not confirmed active. Missing or invalid end dates cannot enter the overdue queue. Group concentrations may reflect workload or stale records; they do not establish misconduct.")

st.write("")

with st.container(border=True):
    st.subheader("Proposals for improvement", anchor="proposals")
    st.caption("Suggested actions for review, not adopted City policy or confirmed enforcement findings.")
    st.dataframe(pd.DataFrame(proposals), use_container_width=True, hide_index=True)

st.write("")

with st.container(border=True):
    st.subheader("AI pattern analysis and proposals", anchor="ai-analysis")
    st.caption("Optional: sends the displayed aggregate statistics and top group names to OpenRouter when you click Generate. No raw descriptions or full CSV are sent. Provider charges may apply.")
    with st.expander("Model and system prompt"):
        if "model_options" not in st.session_state:
            st.session_state.model_options = [DEFAULT_MODEL]
        if st.button("Load OpenRouter model catalog"):
            try:
                st.session_state.model_options = sorted(set([DEFAULT_MODEL] + available_models()))
            except Exception:
                st.warning("Could not load the model catalog. You can enter a model ID below.")
        selected = st.selectbox("OpenRouter model", st.session_state.model_options)
        custom = st.text_input("Custom model ID (optional)", placeholder="provider/model-name")
        model = custom.strip() or selected
        system_prompt = st.text_area("System prompt", PROMPT_PATH.read_text(), height=260)

    has_key = bool(os.getenv("OPENROUTER_API_KEY"))
    if not has_key:
        st.info("Set OPENROUTER_API_KEY in .env.local or the server environment to enable AI analysis. The pattern summary and proposals above work without it.")
    fingerprint = hashlib.sha256(json.dumps([summary, model, system_prompt, modified_ns], sort_keys=True).encode()).hexdigest()
    if st.button("Generate AI accountability analysis", disabled=not len(perm) or not has_key):
        try:
            with st.spinner("Analyzing the filtered queue…"):
                result = generate_analysis(summary, model, system_prompt)
            st.session_state.ai_report = (fingerprint, result, model)
        except Exception:
            st.error("AI analysis failed. Check the API key, model availability, account credit, or connection, then retry.")
    report = st.session_state.get("ai_report")
    if report and report[0] == fingerprint:
        st.caption(f"AI-generated analysis · {report[2]} · review against the statistics above")
        st.markdown(report[1])
        st.download_button("Download AI analysis", report[1], "pittsburgh-analysis.md")
    elif report:
        st.info("The data, filter, model, or prompt changed. Generate a new analysis for this view.")

st.write("")

with st.container(border=True):
    st.subheader("Sources and getting an issue fixed", anchor="sources-and-contact")
    st.markdown("""
    **Data:** This dashboard reads only **pgh data.csv**, the local snapshot of
    [WPRDC’s DOMI Street Closures for GIS Mapping](https://data.wprdc.org/dataset/street-closures).
    The City of Pittsburgh supplies the permit records; WPRDC publishes the dataset and field definitions.
    The dashboard does not refresh from the live feed. Replace the local file to update the snapshot.
    Map background: CARTO / OpenStreetMap (attribution shown on the map).

    **Report a road or sidewalk problem:** Use [Pittsburgh 311](https://www.pittsburghpa.gov/Resident-Services/311)
    for non-emergency City concerns. Include the street and nearest intersection, permit ID if available,
    what is blocked, when you observed it, and photos if safe. Keep your service-request number for follow-up.

    **Ask about a permit, extension, or incorrect closure record:** Contact DOMI at
    [domipermits@pittsburghpa.gov](mailto:domipermits@pittsburghpa.gov) or **412-255-2370**, as listed on the
    [City’s right-of-way applicant guidance](https://www.pittsburghpa.gov/Business-Development/Mobility-and-Infrastructure/Right-of-Way-Management/Applicant-Guidance).
    Ask whether the permit was extended, whether work is complete, and whether its published status/end date needs correction.
    For an immediate emergency, call **911**.
    """)
    st.caption("Snapshot collection date is not supplied in the CSV. File modification time is not evidence of data freshness.")
