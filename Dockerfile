FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_ASSET_OUTPUT_DIR=/app/output

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home appuser && mkdir -p /app/output && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "image_asset_extractor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
