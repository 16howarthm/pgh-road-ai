import logging
from pathlib import Path

import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError

from road_ai import (
    DEFAULT_CATEGORIES,
    classify_one,
    jetstream_client,
    load_fixed_splits,
    make_examples,
)


PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def classification_error_message(exc):
    """Describe actionable failures without exposing credentials or provider bodies."""
    if isinstance(exc, APITimeoutError):
        return "Jetstream took too long to respond. Please try again shortly."
    if isinstance(exc, APIConnectionError):
        return "Cannot connect to Jetstream. Please try again shortly."
    if isinstance(exc, APIStatusError):
        if exc.status_code in (401, 403):
            return "Jetstream rejected API access. Check jetstream_api_key in the app's Secrets settings."
        if exc.status_code == 429:
            return "Jetstream is rate limiting requests. Please try again shortly."
        if exc.status_code >= 500:
            return (
                f"Jetstream is temporarily unavailable (HTTP {exc.status_code}). "
                "Please try again after the service recovers."
            )
        return "Jetstream rejected the classification request. The app administrator should check the model configuration."
    if isinstance(exc, ValueError):
        return "The model did not return a valid category. Please try again."
    return "Classification failed. Please try again."


def classify_description(description, api_key=None, client=None):
    """Classify one description with the evaluated six-example Scout condition."""
    description = str(description).strip()
    if not description:
        raise ValueError("Enter a work description.")

    prompt_dev, _ = load_fixed_splits("work_type", PROJECT_ROOT / "category_splits")
    examples = make_examples(
        prompt_dev, "work_type", n_examples=6, random_state=42
    )
    if len(examples) != 6:
        raise RuntimeError("The six-example production prompt could not be constructed.")

    client = client or jetstream_client(api_key)
    result = classify_one(
        description,
        "llama-4-scout",
        categories=DEFAULT_CATEGORIES,
        examples=examples,
        client=client,
    )
    category = result["category"]
    if category not in DEFAULT_CATEGORIES:
        raise ValueError("The model returned an invalid category.")
    return category


def main():
    st.set_page_config(page_title="Pittsburgh Street Closure Classifier")
    st.title("Pittsburgh Street Closure Classifier")
    st.write(
        "Enter a work description to predict the City of Pittsburgh "
        "street-closure work category."
    )
    description = st.text_area(
        "Work description",
        placeholder="Example: Closing a travel lane while a crane lifts rooftop equipment.",
        height=160,
    )

    if not st.button("Classify", type="primary"):
        return
    if not description.strip():
        st.warning("Enter a work description before classifying.")
        return

    try:
        api_key = st.secrets["jetstream_api_key"]
    except Exception:
        st.error("Jetstream API access is not configured.")
        return

    try:
        with st.spinner("Classifying..."):
            category = classify_description(description, api_key=api_key)
    except Exception as exc:
        # Do not log the exception body: it may contain the submitted description
        # or sensitive provider details. Type and status identify infrastructure failures.
        logger.error(
            "Classification failed: type=%s status=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        st.error(classification_error_message(exc))
        return

    st.subheader("Predicted category")
    st.success(category)
    st.caption("Llama 4 Scout · 6-example prompt")


if __name__ == "__main__":
    main()
