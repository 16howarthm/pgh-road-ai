from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


DEFAULT_CATEGORIES = ["MACHINERY", "DEMOLITION DUMPSTER", "CRANE", "BARRICADE", "MATERIALS", "SCAFFOLD"]
NORMALIZED_DESCRIPTION_COL = "normalized_description"
FAILED_PREDICTION = "__FAILED__"
RESULT_KEY_COLUMNS = ["model", "prompt_condition", "permit_id"]
LLAMA_SCOUT_MODEL = "meta-llama/llama-4-scout"
MODEL_PROVIDER_ROUTING = {
    LLAMA_SCOUT_MODEL: {
        "only": ["google-vertex"],
        "require_parameters": True,
    }
}

__all__ = [
    "DEFAULT_CATEGORIES",
    "NORMALIZED_DESCRIPTION_COL",
    "FAILED_PREDICTION",
    "RESULT_KEY_COLUMNS",
    "LLAMA_SCOUT_MODEL",
    "MODEL_PROVIDER_ROUTING",
    "openrouter_client",
    "dedupe_permits",
    "prepare_eligible_data",
    "balanced_sample",
    "validate_splits",
    "split_prompt_test",
    "save_fixed_splits",
    "load_fixed_splits",
    "make_examples",
    "classification_prompt",
    "classify_one",
    "save_results_checkpoint",
    "load_results_checkpoint",
    "completed_result_keys",
    "remove_model_routing_failures",
    "run_experiment",
    "score_results",
]


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split()).upper()


def _require_columns(df: pd.DataFrame, columns) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _stable_key(seed: int, purpose: str, *values) -> str:
    text = "|".join([str(seed), purpose, *[str(value) for value in values]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def openrouter_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or pass api_key.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def dedupe_permits(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one deterministic observation for each nonempty permit ID."""
    _require_columns(df, ["permit_id", "work_description"])
    out = df.copy()
    out["permit_id"] = out["permit_id"].map(_normalize_text)
    out["work_description"] = out["work_description"].fillna("").astype(str).str.strip()
    out = out[out["permit_id"].ne("")]
    sort_columns = [column for column in ["permit_id", "closure_id", "_id"] if column in out.columns]
    out = out.sort_values(sort_columns, kind="stable")
    return out.drop_duplicates("permit_id", keep="first").reset_index(drop=True)


def prepare_eligible_data(df, label_col, categories=DEFAULT_CATEGORIES):
    """Prepare permit-level eligible data and remove descriptions with conflicting labels."""
    _require_columns(df, [label_col])
    labels = [_normalize_text(category) for category in categories]
    eligible = dedupe_permits(df)
    eligible[label_col] = eligible[label_col].map(_normalize_text)
    eligible[NORMALIZED_DESCRIPTION_COL] = eligible["work_description"].map(_normalize_text)
    eligible = eligible[
        eligible[label_col].isin(labels) & eligible[NORMALIZED_DESCRIPTION_COL].ne("")
    ].copy()

    label_counts = eligible.groupby(NORMALIZED_DESCRIPTION_COL)[label_col].nunique()
    ambiguous_descriptions = set(label_counts[label_counts.gt(1)].index)
    ambiguous_mask = eligible[NORMALIZED_DESCRIPTION_COL].isin(ambiguous_descriptions)
    audit = {
        "eligible_permits_before_ambiguity_filter": int(len(eligible)),
        "ambiguous_description_groups": int(len(ambiguous_descriptions)),
        "ambiguous_records_excluded": int(ambiguous_mask.sum()),
    }
    return eligible.loc[~ambiguous_mask].reset_index(drop=True), audit


def balanced_sample(
    df,
    label_col,
    categories=DEFAULT_CATEGORIES,
    n_per_category=50,
    random_state=42,
    return_audit=False,
):
    """Build a reproducible balanced corpus using unique, unambiguous descriptions."""
    eligible, audit = prepare_eligible_data(df, label_col, categories)
    labels = [_normalize_text(category) for category in categories]

    # Text is the model input, so retain only one permit for each remaining exact
    # normalized description. This prevents a description from crossing splits.
    candidates = eligible.sort_values(
        [NORMALIZED_DESCRIPTION_COL, "permit_id"], kind="stable"
    ).drop_duplicates(NORMALIZED_DESCRIPTION_COL, keep="first")

    parts = []
    for label in labels:
        group = candidates[candidates[label_col].eq(label)].copy()
        if len(group) < n_per_category:
            raise ValueError(
                f"Category {label!r} has {len(group)} unique eligible descriptions; "
                f"{n_per_category} are required."
            )
        group["_sample_order"] = [
            _stable_key(random_state, "sample", label, permit_id, description)
            for permit_id, description in zip(group["permit_id"], group[NORMALIZED_DESCRIPTION_COL])
        ]
        parts.append(group.sort_values("_sample_order").head(n_per_category))

    sample = pd.concat(parts, ignore_index=True).drop(columns="_sample_order")
    sample["_sample_order"] = [
        _stable_key(random_state, "shuffle", permit_id, description)
        for permit_id, description in zip(sample["permit_id"], sample[NORMALIZED_DESCRIPTION_COL])
    ]
    sample = sample.sort_values("_sample_order").drop(columns="_sample_order").reset_index(drop=True)
    audit.update(
        {
            "sample_size": int(len(sample)),
            "records_per_category": int(n_per_category),
            "random_state": int(random_state),
        }
    )
    return (sample, audit) if return_audit else sample


def validate_splits(prompt_df, test_df, label_col, prompt_n=120, test_n=180):
    """Raise when the fixed splits do not satisfy the experiment invariants."""
    for frame in [prompt_df, test_df]:
        _require_columns(frame, ["permit_id", "work_description", label_col])
        if NORMALIZED_DESCRIPTION_COL not in frame:
            frame[NORMALIZED_DESCRIPTION_COL] = frame["work_description"].map(_normalize_text)

    if len(prompt_df) != prompt_n or len(test_df) != test_n:
        raise ValueError(
            f"Expected {prompt_n} prompt-development and {test_n} test rows; "
            f"found {len(prompt_df)} and {len(test_df)}."
        )
    if prompt_df["permit_id"].duplicated().any() or test_df["permit_id"].duplicated().any():
        raise ValueError("Each split must contain unique permit IDs.")
    if prompt_df[NORMALIZED_DESCRIPTION_COL].duplicated().any() or test_df[NORMALIZED_DESCRIPTION_COL].duplicated().any():
        raise ValueError("Each split must contain unique normalized descriptions.")
    if set(prompt_df["permit_id"]) & set(test_df["permit_id"]):
        raise ValueError("Permit IDs overlap between prompt-development and test splits.")
    if set(prompt_df[NORMALIZED_DESCRIPTION_COL]) & set(test_df[NORMALIZED_DESCRIPTION_COL]):
        raise ValueError("Normalized descriptions overlap between prompt-development and test splits.")
    return True


def split_prompt_test(df, label_col, prompt_n=120, test_n=180, random_state=42):
    """Create deterministic, equally stratified prompt-development and test splits."""
    _require_columns(df, ["permit_id", "work_description", label_col])
    data = df.copy()
    if NORMALIZED_DESCRIPTION_COL not in data:
        data[NORMALIZED_DESCRIPTION_COL] = data["work_description"].map(_normalize_text)
    if len(data) != prompt_n + test_n:
        raise ValueError("The balanced corpus must contain exactly prompt_n + test_n rows.")

    labels = sorted(data[label_col].map(_normalize_text).unique())
    if prompt_n % len(labels) or test_n % len(labels):
        raise ValueError("Split sizes must be divisible by the number of categories.")
    prompt_per_label, test_per_label = prompt_n // len(labels), test_n // len(labels)

    prompt_parts, test_parts = [], []
    for label in labels:
        group = data[data[label_col].map(_normalize_text).eq(label)].copy()
        if len(group) != prompt_per_label + test_per_label:
            raise ValueError(f"Category {label!r} does not have the required balanced count.")
        group["_split_order"] = [
            _stable_key(random_state, "split", permit_id, description)
            for permit_id, description in zip(group["permit_id"], group[NORMALIZED_DESCRIPTION_COL])
        ]
        group = group.sort_values("_split_order").drop(columns="_split_order")
        prompt_parts.append(group.iloc[:prompt_per_label])
        test_parts.append(group.iloc[prompt_per_label:])

    prompt = pd.concat(prompt_parts, ignore_index=True).sort_values("permit_id").reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True).sort_values("permit_id").reset_index(drop=True)
    validate_splits(prompt, test, label_col, prompt_n, test_n)
    return prompt, test


def save_fixed_splits(prompt_df, test_df, label_col, output_dir, audit=None):
    """Persist the minimal fixed records needed to reproduce every experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["permit_id", "work_description", label_col, NORMALIZED_DESCRIPTION_COL]
    prompt_path = output_dir / "prompt_development.csv"
    test_path = output_dir / "heldout_test.csv"
    prompt_df[columns].to_csv(prompt_path, index=False)
    test_df[columns].to_csv(test_path, index=False)
    if audit is not None:
        with (output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
    return prompt_path, test_path


def load_fixed_splits(label_col, output_dir, prompt_n=120, test_n=180):
    output_dir = Path(output_dir)
    prompt = pd.read_csv(output_dir / "prompt_development.csv")
    test = pd.read_csv(output_dir / "heldout_test.csv")
    validate_splits(prompt, test, label_col, prompt_n, test_n)
    return prompt, test


def make_examples(prompt_df, label_col, n_examples=1, random_state=42):
    """Select exactly n_examples total from the fixed prompt-development set."""
    if n_examples < 0 or n_examples > len(prompt_df):
        raise ValueError("n_examples must be between zero and the prompt-development size.")
    if n_examples == 0:
        return []

    data = prompt_df.copy()
    data["_example_order"] = [
        _stable_key(random_state, "example", permit_id, description)
        for permit_id, description in zip(data["permit_id"], data["work_description"])
    ]
    data = data.sort_values("_example_order")

    # For the several-example condition, cover categories before adding extras.
    if n_examples >= data[label_col].nunique():
        selected = data.groupby(label_col, sort=True, group_keys=False).head(1)
        remaining = data.loc[~data.index.isin(selected.index)]
        selected = pd.concat([selected, remaining.head(n_examples - len(selected))])
    else:
        selected = data.head(n_examples)
    selected = selected.sort_values("_example_order").head(n_examples)
    return [
        {"description": row.work_description, "category": _normalize_text(row[label_col])}
        for _, row in selected.iterrows()
    ]


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
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": classification_prompt(categories, examples)},
            {"role": "user", "content": str(description)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if model in MODEL_PROVIDER_ROUTING:
        request["extra_body"] = {"provider": MODEL_PROVIDER_ROUTING[model]}

    resp = client.chat.completions.create(
        **request,
    )
    raw = resp.choices[0].message.content
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw or "", re.S)
        obj = json.loads(match.group(0)) if match else {}

    category = _normalize_text(obj.get("category"))
    labels = {_normalize_text(value) for value in categories}
    if not category:
        raise ValueError("Model response is missing a category.")
    if category not in labels:
        raise ValueError(f"Out-of-vocabulary category: {category}")
    obj["category"] = category
    obj["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", None)
    obj["completion_tokens"] = getattr(resp.usage, "completion_tokens", None)
    return obj


def save_results_checkpoint(results, path):
    """Atomically save one row per model/prompt/permit experiment key."""
    _require_columns(results, RESULT_KEY_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        results.drop_duplicates(RESULT_KEY_COLUMNS, keep="last")
        .sort_values(RESULT_KEY_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )
    temporary_path = path.with_name(f".{path.name}.tmp")
    checkpoint.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)
    return checkpoint


def load_results_checkpoint(path, models, prompt_conditions, permit_ids):
    """Load and validate resumable results for the configured experiment."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    results = pd.read_csv(path)
    _require_columns(results, [*RESULT_KEY_COLUMNS, "n_examples"])
    results["model"] = results["model"].astype(str)
    results["prompt_condition"] = results["prompt_condition"].astype(str)
    results["permit_id"] = results["permit_id"].astype(str)

    unknown_models = sorted(set(results["model"]) - set(models))
    unknown_conditions = sorted(set(results["prompt_condition"]) - set(prompt_conditions))
    unknown_permits = sorted(set(results["permit_id"]) - {str(value) for value in permit_ids})
    if unknown_models or unknown_conditions or unknown_permits:
        raise ValueError(
            "Checkpoint does not match this experiment: "
            f"models={unknown_models}, conditions={unknown_conditions}, permits={unknown_permits[:5]}"
        )

    expected_examples = results["prompt_condition"].map(prompt_conditions)
    recorded_examples = pd.to_numeric(results["n_examples"], errors="coerce")
    if not recorded_examples.eq(expected_examples).all():
        raise ValueError("Checkpoint example counts do not match the configured prompt conditions.")

    return (
        results.drop_duplicates(RESULT_KEY_COLUMNS, keep="last")
        .sort_values(RESULT_KEY_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )


def completed_result_keys(results):
    if results.empty:
        return set()
    _require_columns(results, RESULT_KEY_COLUMNS)
    return set(results[RESULT_KEY_COLUMNS].itertuples(index=False, name=None))


def remove_model_routing_failures(results, model):
    """Remove only saved 404/no-endpoint failures for one model so they can be retried."""
    if results.empty:
        return results.copy(), 0
    _require_columns(results, ["model", "predicted", "error"])
    errors = results["error"].fillna("").astype(str)
    retry_mask = (
        results["model"].astype(str).eq(str(model))
        & results["predicted"].astype(str).eq(FAILED_PREDICTION)
        & errors.str.contains("Error code: 404", case=False, regex=False)
        & errors.str.contains("No endpoint", case=False, regex=False)
    )
    return results.loc[~retry_mask].reset_index(drop=True), int(retry_mask.sum())


def run_experiment(
    test_df,
    label_col,
    model,
    categories=DEFAULT_CATEGORIES,
    examples=None,
    sleep_s=0.0,
    client=None,
    on_result=None,
    progress_callback=None,
):
    rows = []
    client = client or openrouter_client()
    labels = {_normalize_text(value) for value in categories}
    for _, row in test_df.iterrows():
        result = {
            "permit_id": row.get("permit_id"),
            "description": row.work_description,
            "actual": _normalize_text(row[label_col]),
            "predicted": FAILED_PREDICTION,
            "confidence": None,
            "reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": None,
        }
        try:
            answer = classify_one(row.work_description, model, categories, examples, client)
            prediction = _normalize_text(answer.get("category"))
            if prediction not in labels:
                raise ValueError(f"Invalid prediction: {prediction or '<missing>'}")
            result.update(
                {
                    "predicted": prediction,
                    "confidence": answer.get("confidence"),
                    "reason": answer.get("reason"),
                    "prompt_tokens": answer.get("prompt_tokens"),
                    "completion_tokens": answer.get("completion_tokens"),
                }
            )
        except Exception as exc:
            result["error"] = str(exc)
        rows.append(result)
        if on_result:
            on_result(result.copy())
        if progress_callback:
            progress_callback()
        if sleep_s:
            time.sleep(sleep_s)
    return pd.DataFrame(rows)


def score_results(results, categories=DEFAULT_CATEGORIES):
    """Score every held-out row, mapping invalid or missing outputs to failure."""
    if results.empty:
        raise ValueError("Cannot score empty results.")
    _require_columns(results, ["actual", "predicted"])
    labels = [_normalize_text(value) for value in categories]
    scored = results.copy()
    scored["actual"] = scored["actual"].map(_normalize_text)
    unknown_actual = sorted(set(scored["actual"]) - set(labels))
    if unknown_actual:
        raise ValueError(f"Unknown actual labels: {unknown_actual}")
    scored["predicted"] = scored["predicted"].map(_normalize_text)
    scored.loc[~scored["predicted"].isin(labels), "predicted"] = FAILED_PREDICTION
    matrix_labels = labels + [FAILED_PREDICTION]
    failures = int(scored["predicted"].eq(FAILED_PREDICTION).sum())
    return {
        "accuracy": accuracy_score(scored.actual, scored.predicted),
        "macro_f1": f1_score(
            scored.actual, scored.predicted, labels=labels, average="macro", zero_division=0
        ),
        "classification_report": classification_report(
            scored.actual, scored.predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(scored.actual, scored.predicted, labels=matrix_labels),
            index=matrix_labels,
            columns=matrix_labels,
        ),
        "n_scored": len(scored),
        "n_failures": failures,
        "n_errors": failures,
    }
