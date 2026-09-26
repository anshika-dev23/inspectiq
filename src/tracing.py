"""LangFuse tracing (step 5): one trace per question, with nested spans.

Thin wrapper so the application code does not depend on LangFuse being configured: without keys, or with
TRACING_ENABLED=0 (tests), every call is a no-op. Spans nest through OpenTelemetry context, so a span opened
inside another becomes its child.
"""
import logging
import os
import ssl
from contextlib import contextmanager
from functools import lru_cache

logger = logging.getLogger("inspectiq.tracing")


def ensure_ca_bundle() -> None:
    """python.org builds of Python have no CA file until "Install Certificates.command" is run. requests uses
    certifi and works, but the OpenTelemetry span exporter then fails with CERTIFICATE_VERIFY_FAILED. When no
    CA file is configured and the default one is missing, point OpenSSL at certifi's bundle."""
    default_cafile = ssl.get_default_verify_paths().openssl_cafile
    if "SSL_CERT_FILE" not in os.environ and not os.path.exists(default_cafile):
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        logger.info("no CA file at %s: using certifi's bundle for TLS", default_cafile)


@lru_cache(maxsize=1)
def get_langfuse():
    """The LangFuse client, or None when tracing is off."""
    if os.getenv("TRACING_ENABLED", "1") == "0":
        return None
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("LangFuse keys not set: tracing off")
        return None
    from langfuse import Langfuse

    ensure_ca_bundle()
    return Langfuse(host=os.getenv("LANGFUSE_HOST"))  # keys are read from the environment


@contextmanager
def observe(name: str, as_type: str = "span", **fields):
    """Open a span (as_type: span, chain, retriever, generation, guardrail, ...); yields None when tracing is off."""
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **fields) as observation:
        yield observation


def update(observation, **fields) -> None:
    if observation is not None:
        observation.update(**fields)


def set_trace_io(observation, **fields) -> None:
    """Input/output shown for the whole trace in the LangFuse trace list."""
    if observation is not None:
        observation.set_trace_io(**fields)


def current_trace_url() -> str | None:
    client = get_langfuse()
    if client is None:
        return None
    return client.get_trace_url(trace_id=client.get_current_trace_id())


def flush() -> None:
    """Send buffered spans; call before a script exits."""
    client = get_langfuse()
    if client is not None:
        client.flush()
