ARG UPSTREAM=ghost-racer-upstream:a3495a384
FROM ${UPSTREAM} AS source
USER root
ARG BETAFLIGHT_REF
WORKDIR /tmp/betaflight
# The commit records exact submodule revisions. Only hydrate modules used by
# the upstream build, avoiding the unrelated embedded hardware SDKs.
RUN git init && git remote add origin https://github.com/betaflight/betaflight.git && \
    git fetch --depth=1 origin "$BETAFLIGHT_REF" && git checkout --detach FETCH_HEAD && \
    git submodule update --init --depth=1 src/config lib/modules/mavlink lib/modules/dronecan/libcanard
# Run the FC on the Docker VM's native CPU. Static linking allows an arm64
# binary to use the same loopback network as the official amd64 Gazebo image.
FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git xxd python3 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=source /tmp/betaflight /tmp/betaflight
WORKDIR /tmp/betaflight
COPY sitl-clock.patch /tmp/sitl-clock.patch
COPY sitl-step.patch /tmp/sitl-step.patch
RUN git apply /tmp/sitl-clock.patch && git apply /tmp/sitl-step.patch && make -j4 CONFIG=SITL_GAZEBO OPTIONS=SITL_STATIC LTO=no && \
    install -m 0755 obj/main/betaflight_SITL_SITL_GAZEBO.elf /usr/local/bin/betaflight-sitl

FROM ${UPSTREAM}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gz-transport13 python3-gz-msgs10 python3-pil libgl1-mesa-dri && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -c 'from gz.transport13 import Node; from gz.msgs10.image_pb2 import Image; from PIL import Image as PILImage'
COPY rotors /tmp/whoop-rotors
RUN cmake -S /tmp/whoop-rotors -B /tmp/whoop-build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/whoop-build -j2 && install /tmp/whoop-build/libWhoopRotors.so /usr/local/lib/
COPY --from=build /usr/local/bin/betaflight-sitl /usr/local/bin/betaflight-sitl
USER devpod
WORKDIR /workspace
