# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    WEBRETRIEVER_MAX_STEPS=100

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 webagent \
    && useradd --uid 10001 --gid webagent --create-home --shell /usr/sbin/nologin webagent \
    && mkdir -p /work/input /work/output \
    && chown -R webagent:webagent /work

COPY pyproject.toml README.md README_zh.md THIRD_PARTY_NOTICES.md ./
COPY src ./src
COPY vendor ./vendor

# 正式镜像只安装 Playwright 客户端，不下载或启动浏览器。
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

USER webagent

HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD ["python", "-m", "web_agent.cli", "--healthcheck", "--output", "/work/output"]

ENTRYPOINT ["python", "-m", "web_agent.cli"]
CMD ["--healthcheck", "--output", "/work/output"]
