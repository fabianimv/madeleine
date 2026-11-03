# madeleine MCP server (stdio by default).
#   docker build -t madeleine-mcp .
#   docker run -i --rm -v madeleine-data:/data madeleine-mcp            # MCP over stdio
#   docker run --rm madeleine-mcp demo                                   # offline demo
#   docker run --rm -p 8765:8765 -v madeleine-data:/data \
#     -e MADELEINE_API_TOKEN=change-me -e MADELEINE_ALLOWED_HOSTS=localhost \
#     madeleine-mcp serve --transport http --host 0.0.0.0              # REST + MCP over HTTP
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MADELEINE_DB=/data/madeleine.db

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 10001 madeleine \
    && mkdir -p /data && chown madeleine:madeleine /data && chmod 700 /data

USER madeleine
VOLUME ["/data"]
EXPOSE 8765
ENTRYPOINT ["madeleine-mcp"]
CMD ["serve"]
