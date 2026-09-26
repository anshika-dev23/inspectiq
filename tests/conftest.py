import os

# Tests never send traces to LangFuse (src/tracing.py reads this before creating a client).
os.environ["TRACING_ENABLED"] = "0"
