FROM gocv/opencv:4.10.0-ubuntu-22.04 as builder

ENV NAME=rs-image-processing-service

RUN set -xeu && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile=minimal

ENV PATH="${PATH}:/root/.cargo/bin"

RUN apt update
RUN apt install -y clang libclang-dev
RUN apt install -y protobuf-compiler
RUN apt install -y

WORKDIR /opt/app
RUN cargo new --bin ${NAME}
WORKDIR  /opt/app/${NAME}
COPY . .
RUN cargo build --release

FROM gocv/opencv:4.10.0-ubuntu-22.04

RUN mkdir /opt/app
WORKDIR  /opt/app

RUN apt update
RUN apt install -y libjemalloc2

RUN export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so

ENV NAME=rs-image-processing-service

# copy the binary into the final image
COPY --from=builder  /opt/app/${NAME}/target/release/${NAME} /opt/app/

RUN mkdir ./conf
RUN touch ./conf/config.toml

CMD ["./rs-image-processing-service"]