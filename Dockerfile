FROM python:3.10-slim-bookworm

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    dssp \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. DSSP Setup - Create proper wrapper for Biopython compatibility
RUN echo "Setting up DSSP..." && \
    mkdssp --version && \
    MKDSSP_PATH=$(which mkdssp) && \
    rm -f /usr/bin/dssp && \
    printf '#!/bin/bash\n# Wrapper to call mkdssp with legacy output format\nexec %s --output-format=dssp "$@"\n' "${MKDSSP_PATH}" > /usr/bin/dssp && \
    chmod +x /usr/bin/dssp && \
    echo "DSSP wrapper created successfully" && \
    cat /usr/bin/dssp

# 3. Install KPAX (Handles local tarball)
WORKDIR /tmp/kpax
COPY kpax.tar.gz .
RUN tar -xzvf kpax.tar.gz --no-same-owner --strip-components=1 && \
    if [ -f Makefile ]; then make; else echo "No Makefile found, assuming binary..."; fi && \
    find . -type f \( -name "Kpax" -o -name "kpax" \) -exec cp {} /usr/local/bin/kpax \; && \
    chmod +x /usr/local/bin/kpax

WORKDIR /app

# 4. Dependencies
COPY requirements.txt .
RUN sed -i 's/Bio==1.3.2/biopython==1.81/' requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit stmol shimmy ipython_genutils

COPY . .
RUN pip install .

RUN mkdir -p /app/data_temp/input /app/data_temp/output
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]