FROM python:3.10-slim

# Update and install required packages including clang
RUN apt-get update && apt-get install -y \
    apt-utils \
    vim \
    telnet \
    iputils-ping \
    curl \
    libgl1 \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    && apt-get clean

# Create working directory
RUN mkdir -p /usr/local/bin/cde
WORKDIR /usr/local/bin/cde

# Copy requirements
COPY requirements.txt requirements.txt

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Python packages
RUN python -m nltk.downloader punkt