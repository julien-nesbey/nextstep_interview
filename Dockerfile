# --- Python Builder Stage ---
FROM python:3.12-slim AS python-builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  python3-dev \
  gcc \
  g++ \
  libatlas-base-dev \
  meson \
  ninja-build \
  && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install base dependencies first
RUN pip install -U pip setuptools wheel

# Copy requirements file
COPY src/interview_app/requirements.txt .

# Install Python packages with size optimization
RUN pip install --no-cache-dir -r requirements.txt && \
  # Remove downloaded model caches to save space
  rm -rf /root/.cache/huggingface && \
  rm -rf /root/.cache/pip

# --- Final Python Runtime Stage ---
FROM python:3.12-slim

WORKDIR /app

# Install only the runtime dependencies (not build dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libstdc++6 \
  libgomp1 \
  curl \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Copy virtual environment and Python app
COPY --from=python-builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Only copy required files, not the entire directory
COPY src/interview_app/*.py ./src/interview_app/
COPY src/interview_app/models ./src/interview_app/models
COPY src/interview_app/prompts ./src/interview_app/prompts
COPY src/interview_app/utils ./src/interview_app/utils
COPY .env* ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Set this to prevent models from being downloaded to cache
ENV TRANSFORMERS_OFFLINE=1
# Prevent unnecessary model caching
ENV SENTENCE_TRANSFORMERS_HOME=/app/src/interview_app/models

EXPOSE 5000

CMD ["/venv/bin/python", "src/interview_app/app.py"]