# PulseDiscover V2 — production-SHAPED serving image (NOT a production deployment artifact)
FROM python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] faiss-cpu numpy pandas scipy scikit-learn

# code + model/index inputs (data mounted at runtime or baked in)
COPY src/ /app/src/
ENV PYTHONPATH=/app/src
ENV PD_DATADIR=/app/data/interim
ENV PD_DEFAULT_MODE=exact

EXPOSE 8000
# startup loads ALS factors + builds FAISS FlatIP (default) + HNSW (scale mode), then serves
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
