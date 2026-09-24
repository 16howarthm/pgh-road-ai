from __future__ import annotations
import json, os, re, time
from dataclasses import dataclass
from typing import Iterable, Optional
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

DEFAULT_CATEGORIES = ["MACHINERY", "DEMOLITION DUMPSTER", "CRANE", "BARRICADE", "MATERIALS", "SCAFFOLD"]


def openrouter_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or pass api_key.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def dedupe_permits(df: pd.DataFrame) -> pd.DataFrame:
    """One record per permit/description so repeated GIS segments cannot leak across splits."""
    out = df.copy()
    out["work_description"] = out["work_description"].fillna("").astype(str).str.strip()
    keys = [c for c in ["permit_id", "work_description"] if c in out.columns]
    return out.drop_duplicates(keys).reset_index(drop=True)


def balanced_sample(df, label_col, categories=DEFAULT_CATEGORIES, n_per_category=50, random_state=42):
    d = dedupe_permits(df)
    d[label_col] = d[label_col].astype(str).str.upper().str.strip()
    d = d[d[label_col].isin([x.upper() for x in categories]) & d.work_description.ne("")]
    parts=[]
    for cat, g in d.groupby(label_col):
        parts.append(g.sample(min(n_per_category, len(g)), random_state=random_state))
    if not parts:
        raise ValueError(f"No rows match categories in {label_col}. Check the full dataset/label column.")
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def split_prompt_test(df, label_col, prompt_n=120, test_n=180, random_state=42):
    """Stratified split after permit/description de-duplication."""
    from sklearn.model_selection import train_test_split
    if prompt_n + test_n > len(df):
        raise ValueError("Requested split is larger than the sample.")
    if prompt_n + test_n == len(df):
        temp = df
    else:
        temp, _ = train_test_split(df, train_size=prompt_n+test_n, stratify=df[label_col], random_state=random_state)
    prompt, test = train_test_split(temp, train_size=prompt_n, test_size=test_n, stratify=temp[label_col], random_state=random_state)
    return prompt.reset_index(drop=True), test.reset_index(drop=True)


def make_examples(prompt_df, label_col, k_per_class=1, random_state=42):
    if k_per_class <= 0: return []
    rows=[]
    for _, g in prompt_df.groupby(label_col):
        for _, r in g.sample(min(k_per_class,len(g)), random_state=random_state).iterrows():
            rows.append({"description": r.work_description, "category": str(r[label_col]).upper()})
    return rows


def classification_prompt(categories, examples=None):
    examples = examples or []
    ex = "\n".join(f'Description: {e["description"]}\nCategory: {e["category"]}' for e in examples)
    example_section = "Labeled examples:\n" + ex if ex else ""
    return f"""You classify Pittsburgh street-closure work descriptions into exactly one allowed construction-staging category.
Allowed categories: {', '.join(categories)}.
Use only evidence in the description. Do not invent facts. If wording is ambiguous, choose the single best allowed category.
Return valid JSON only: {{"category":"<allowed category>","confidence":0.0,"reason":"<brief evidence-based reason>"}}.
Confidence must be between 0 and 1.
{example_section}"""


def classify_one(description, model, categories=DEFAULT_CATEGORIES, examples=None, client=None):
    client = client or openrouter_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":classification_prompt(categories, examples)},
                  {"role":"user","content":str(description)}],
        response_format={"type":"json_object"}, temperature=0,
    )
    raw = resp.choices[0].message.content
    try: obj=json.loads(raw)
    except Exception:
        m=re.search(r"\{.*\}", raw or "", re.S); obj=json.loads(m.group(0)) if m else {}
    obj["category"] = str(obj.get("category","")).upper().strip()
    obj["prompt_tokens"] = getattr(resp.usage,"prompt_tokens",None)
    obj["completion_tokens"] = getattr(resp.usage,"completion_tokens",None)
    return obj


def run_experiment(test_df, label_col, model, categories=DEFAULT_CATEGORIES, examples=None, sleep_s=0.0):
    rows=[]; client=openrouter_client()
    for i,r in test_df.iterrows():
        try:
            ans=classify_one(r.work_description, model, categories, examples, client)
            rows.append({"permit_id":r.get("permit_id"), "description":r.work_description,
                         "actual":str(r[label_col]).upper(), "predicted":ans.get("category"),
                         "confidence":ans.get("confidence"), "reason":ans.get("reason"),
                         "prompt_tokens":ans.get("prompt_tokens"), "completion_tokens":ans.get("completion_tokens"), "error":None})
        except Exception as e:
            rows.append({"permit_id":r.get("permit_id"), "description":r.work_description,
                         "actual":str(r[label_col]).upper(), "predicted":None, "error":str(e)})
        if sleep_s: time.sleep(sleep_s)
    return pd.DataFrame(rows)


def score_results(results, categories=DEFAULT_CATEGORIES):
    ok=results.dropna(subset=["predicted"]).copy()
    labels=[x.upper() for x in categories]
    return {
        "accuracy": accuracy_score(ok.actual, ok.predicted),
        "macro_f1": f1_score(ok.actual, ok.predicted, labels=labels, average="macro", zero_division=0),
        "classification_report": classification_report(ok.actual, ok.predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": pd.DataFrame(confusion_matrix(ok.actual,ok.predicted,labels=labels),index=labels,columns=labels),
        "n_scored":len(ok), "n_errors":len(results)-len(ok)
    }
