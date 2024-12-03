# Stage 1: Install dependencies
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Copy only the requirements file to avoid unnecessary cache invalidation
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Copy the app and dependencies to a smaller image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Copy the application code
COPY . .

# Ensure the virtual environment is used
ENV PATH="/venv/bin:$PATH"

# Set the default command to run your main.py script
CMD ["python", "main.py"]
