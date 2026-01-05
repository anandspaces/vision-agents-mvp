# =========================
# Builder stage
# =========================
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required to build wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency metadata only (for caching)
COPY pyproject.toml README.md ./

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .


# =========================
# Runtime stage
# =========================
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime-only system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install from wheels (fast + reproducible)
RUN pip install --no-cache-dir /wheels/*

# Copy application code (dockerignore will exclude junk)
COPY . .

# Expose FastAPI port
EXPOSE 9040

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9040"]
