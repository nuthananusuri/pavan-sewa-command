# Multi-arch image: builds natively on s390x (IBM LinuxONE) and on x86_64 / arm64.
#   podman build -f deploy/Dockerfile -t pavan-sewa-command .
#   podman run -p 8080:8080 -v psc-data:/data pavan-sewa-command
FROM docker.io/library/python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PSC_DB=/data/pavan_sewa.db \
    PSC_KEY_FILE=/data/.psc_key \
    PSC_PORT=8080
RUN mkdir -p /data && useradd -r -u 10001 psc && chown -R psc /data /app
USER psc
EXPOSE 8080
CMD ["python", "app.py"]
