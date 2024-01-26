FROM python:3.8-alpine as builder

WORKDIR /app

RUN python3 -m venv venv
ENV PATH="/app/venv/bin:${PATH}"

WORKDIR /app/crynux-chat

COPY src src
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && pip install .

FROM python:3.8-alpine as final

COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:${PATH}"

WORKDIR /app

COPY assets assets
COPY start.sh start.sh
COPY config.yml.example config.yml.example

ENV CRYNUX_CHAT_CONFIG="/app/config/config.yml"

ENTRYPOINT ["bash", "start.sh"]
