# --- Build stage ---
FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir hatchling
COPY pyproject.toml .
COPY kintsugi/ kintsugi/
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# --- Runtime stage ---
FROM python:3.12-slim
WORKDIR /app

RUN addgroup --system kintsugi && adduser --system --ingroup kintsugi kintsugi

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY alembic.ini .
COPY migrations/ migrations/
COPY kintsugi/ kintsugi/

USER kintsugi
EXPOSE 8000

CMD ["uvicorn", "kintsugi.main:app", "--host", "0.0.0.0", "--port", "8000"]
