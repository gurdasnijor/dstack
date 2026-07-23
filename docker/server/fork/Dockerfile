FROM python:3.11-slim

ARG GIT_SHA
ENV DSTACK_FORK_GIT_SHA=$GIT_SHA
ENV PYTHONUNBUFFERED=1
ENV DSTACK_SERVER_LOG_FORMAT=json

WORKDIR /dstack-server

RUN apt-get update && apt-get install -y \
    curl \
    git \
    sqlite3 \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$(uname -m)" = "aarch64" ]; then ARCH="arm64"; else ARCH="x86_64"; fi && \
    curl -fsSLO "https://github.com/benbjohnson/litestream/releases/download/v0.5.9/litestream-0.5.9-linux-$ARCH.deb" && \
    dpkg -i "litestream-0.5.9-linux-$ARCH.deb" && \
    rm "litestream-0.5.9-linux-$ARCH.deb"

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY skills ./skills
RUN uv tool install --no-cache '.[all]'

COPY docker/server/entrypoint.sh ./entrypoint.sh
RUN chmod 755 ./entrypoint.sh

EXPOSE 3000

ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]
