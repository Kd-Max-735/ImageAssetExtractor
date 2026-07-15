FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/RMBG20

WORKDIR /service
COPY docker/rmbg-requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY docker/rmbg_service.py ./rmbg_service.py

EXPOSE 8000
CMD ["uvicorn", "rmbg_service:app", "--host", "0.0.0.0", "--port", "8000"]

