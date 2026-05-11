FROM rust:1.83-slim AS builder
WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/vjepa-risk-gateway /usr/local/bin/vjepa-risk-gateway
EXPOSE 8080
CMD ["vjepa-risk-gateway"]
