FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POWER_ECON_CONFIG=/app/configs/demo.yaml

WORKDIR /app
COPY . /app

ARG PIP_EXTRAS=""
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -n "$PIP_EXTRAS" ]; then pip install -e ".[${PIP_EXTRAS}]"; else pip install -e .; fi

EXPOSE 8000
CMD ["power-econ", "serve", "-c", "/app/configs/demo.yaml", "--host", "0.0.0.0", "--port", "8000"]
