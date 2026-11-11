FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs
COPY knowledge ./knowledge
RUN pip install --no-cache-dir -e . shap fastapi uvicorn anthropic
# Model registry is mounted (or baked in by CI after `co2-ml train`)
VOLUME /app/models
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "co2_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
