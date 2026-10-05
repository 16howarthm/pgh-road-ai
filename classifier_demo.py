import logging
import time
from pathlib import Path

import streamlit as st

from road_ai import (
    DEFAULT_CATEGORIES,
    classify_one,
    jetstream_client,
    load_fixed_splits,
    make_examples,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = "llama-4-scout"
MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (1, 2)
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


class DemoClassificationError(Exception):
    """A safe, user-facing classifier failure."""

    user_message = "Classification failed. Please try again."


class JetstreamAccessError(DemoClassificationError):
    user_message = "Jetstream rejected API access. Check the app's Jetstream secret."


class JetstreamRateLimitError(DemoClassificationError):
    user_message = (
        "Jetstream is temporarily rate limiting requests. Please try again shortly."
    )


class JetstreamUnavailableError(DemoClassificationError):
    user_message = "Jetstream is temporarily unavailable. Please try again shortly."


class InvalidModelOutputError(DemoClassificationError):
    user_message = "The model did not return a valid category. Please try again."


class ClassifierConfigurationError(DemoClassificationError):
    user_message = "The classifier is not configured correctly. Please contact the app owner."


def _http_status(exc):
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status


def _is_connection_or_timeout(exc):
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    exception_types = {cls.__name__ for cls in type(exc).__mro__}
    connection_types = {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
    }
    return not exception_types.isdisjoint(connection_types)


def _log_attempt_failure(exc, attempt):
    logger.warning(
        "Jetstream classification attempt failed: exception_type=%s "
        "http_status=%s attempt=%d",
        type(exc).__name__,
        _http_status(exc),
        attempt,
    )


def _demo_jetstream_client(api_key):
    # The demo owns its bounded retry policy. Disabling SDK retries ensures that
    # MAX_ATTEMPTS is also the maximum number of HTTP requests.
    return jetstream_client(api_key).with_options(max_retries=0)


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

    client = client or _demo_jetstream_client(api_key)
    malformed_failures = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = classify_one(
                description,
                MODEL,
                categories=DEFAULT_CATEGORIES,
                examples=examples,
                client=client,
            )
            category = result["category"]
            if category not in DEFAULT_CATEGORIES:
                raise ValueError("The model returned an invalid category.")
            return category
        except Exception as exc:
            status = _http_status(exc)
            _log_attempt_failure(exc, attempt)

            if status in {401, 403}:
                raise JetstreamAccessError from None

            if status in TRANSIENT_HTTP_STATUSES or _is_connection_or_timeout(exc):
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
                    continue
                if status == 429:
                    raise JetstreamRateLimitError from None
                raise JetstreamUnavailableError from None

            # classify_one uses ValueError only for missing, malformed, or
            # out-of-vocabulary model output. Retry that failure at most once.
            if isinstance(exc, (ValueError, KeyError, TypeError, AttributeError)):
                malformed_failures += 1
                if malformed_failures < 2 and attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
                    continue
                raise InvalidModelOutputError from None

            raise ClassifierConfigurationError from None

    raise JetstreamUnavailableError


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
    except DemoClassificationError as exc:
        st.error(exc.user_message)
        return
    except Exception as exc:
        logger.error(
            "Classifier setup failed: exception_type=%s http_status=%s",
            type(exc).__name__,
            _http_status(exc),
        )
        st.error(ClassifierConfigurationError.user_message)
        return

    st.subheader("Predicted category")
    st.success(category)
    st.caption("Llama 4 Scout · 6-example prompt")


if __name__ == "__main__":
    main()
