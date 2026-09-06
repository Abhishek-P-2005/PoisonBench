FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Preserve the "api" package structure (api/config.py, api/pipeline/, ...)
# under /app/api/, and integration/ alongside it -- every module in this
# project imports as "from api.xxx import ..." / "from integration.xxx import ...",
# so /app must be the import root, not /app itself flattened.
COPY api/ /app/api/
COPY integration/ /app/integration/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
