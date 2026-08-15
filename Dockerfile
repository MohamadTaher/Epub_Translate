# Build the browser app, then serve it from the Python app that does the work.
FROM node:20-alpine AS web

WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch the tokenizer at build time. Left to first use it would stall the first
# request, and if the host blocks egress it silently falls back to a rough
# character count that skews every estimate.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY epub_translate/ ./epub_translate/
COPY server/ ./server/
COPY translate_epub.py .
COPY --from=web /web/dist ./web/dist

# Uploads and the budget database.
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 7860
# Cloud Run tells the container which port to serve through PORT; 7860 is the
# local default, which is what compose maps. `exec` keeps uvicorn as PID 1 so it
# still receives the shutdown signal rather than the shell swallowing it.
CMD exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}
