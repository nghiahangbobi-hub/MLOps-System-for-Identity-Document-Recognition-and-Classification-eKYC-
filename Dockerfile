FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/pycache HOME=/tmp
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

FROM base AS app
COPY azure/ azure/
COPY data_pipeline/ data_pipeline/
COPY demo/ demo/
COPY labeling/ labeling/
COPY modeling/ modeling/

CMD ["python", "-m", "demo.server", "--host", "0.0.0.0"]

FROM base AS azure-cli
ENV PATH=/opt/azure-cli/bin:$PATH AZURE_EXTENSION_DIR=/opt/azure-extensions
RUN python -m venv /opt/azure-cli \
    && /opt/azure-cli/bin/python -m pip install --no-cache-dir azure-cli==2.93.0 \
    && AZURE_CONFIG_DIR=/tmp/azure-build az extension add --name ml --version 2.45.2 --yes \
    && rm -rf /tmp/azure-build
RUN AZURE_CONFIG_DIR=/tmp/azure-build az extension add --name log-analytics --version 1.0.0b3 --yes \
    && rm -rf /tmp/azure-build

FROM app AS azure
COPY --from=azure-cli /opt/azure-cli /opt/azure-cli
COPY --from=azure-cli /opt/azure-extensions /opt/azure-extensions
ENV PATH=$PATH:/opt/azure-cli/bin AZURE_EXTENSION_DIR=/opt/azure-extensions
