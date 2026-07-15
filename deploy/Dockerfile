# Reason Ready — container image (multi-stage, minimal runtime).
# Build:  docker build -f deploy/Dockerfile -t rrf .
# Run:    docker run -p 7878:7878 -v rrf-estate:/var/lib/rrf rrf
FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends clang libclang-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release --bin rrf --bin rrf-bench

FROM debian:bookworm-slim
RUN useradd --system --home /var/lib/rrf rrf && mkdir -p /var/lib/rrf && chown rrf /var/lib/rrf
COPY --from=build /src/target/release/rrf /src/target/release/rrf-bench /usr/local/bin/
USER rrf
ENV RRF_ESTATE=/var/lib/rrf/estate \
    RRF_LISTEN=0.0.0.0:7878 \
    RRF_EVENTS=/var/lib/rrf/events.jsonl \
    RUST_LOG=info
EXPOSE 7878
VOLUME /var/lib/rrf
ENTRYPOINT ["/usr/local/bin/rrf"]
