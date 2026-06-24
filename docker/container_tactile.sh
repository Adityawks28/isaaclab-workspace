#!/bin/bash
# Isaac Sim 5.1 (Docker) launcher for the tactile data factory.
# Sibling of container.sh (4.5.0) — separate image + container so the proven 4.5.0
# RL setup stays untouched. Build first: docker build -f docker/Dockerfile.tactile -t isaac-tactile:5.1 .

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LAB_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

CONTAINER_NAME="isaac-tactile"
IMAGE="isaac-tactile:5.1"
CACHE="${HOME}/docker/isaac-tactile"

show_help() {
    echo "Usage: $0 [start|enter|gui|stop|status]"
    echo "Your lab code (${LAB_DIR}) is mounted at /root/lab inside the container."
}

setup_x11() {
    if [ -n "$DISPLAY" ]; then xhost +local:root >/dev/null 2>&1 || true
    else echo "Warning: DISPLAY not set, GUI unavailable."; fi
}

start_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker start "${CONTAINER_NAME}" >/dev/null; echo "Started. Use '$0 enter'."; return
    fi
    setup_x11
    mkdir -p "${CACHE}"/{cache/{kit,ov,pip,glcache,computecache},logs,data,documents}
    docker run --name "${CONTAINER_NAME}" -d --entrypoint bash \
        --runtime=nvidia --gpus all \
        -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" -e "OMNI_KIT_ALLOW_ROOT=1" \
        -e "DISPLAY=${DISPLAY}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "${LAB_DIR}:/root/lab:rw" \
        -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
        -v "${CACHE}/cache/ov:/root/.cache/ov:rw" \
        -v "${CACHE}/cache/pip:/root/.cache/pip:rw" \
        -v "${CACHE}/logs:/root/.nvidia-omniverse/logs:rw" \
        -v "${CACHE}/data:/root/.local/share/ov/data:rw" \
        -v "${CACHE}/documents:/root/Documents:rw" \
        "${IMAGE}" -c "tail -f /dev/null"
    echo "Container running. Use '$0 enter' or '$0 gui'."
}

enter_container() { setup_x11; docker exec -it "${CONTAINER_NAME}" bash; }
gui_container()   { setup_x11; docker exec -it -e OMNI_KIT_ALLOW_ROOT=1 "${CONTAINER_NAME}" \
                      bash -c "cd /isaac-sim && ./isaac-sim.sh --allow-root"; }
stop_container()  { docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 && echo "Removed." || echo "Not present."; }
status_container(){ docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$" && echo "Running." || echo "Not running."; }

case "$1" in
    start) start_container ;; enter) enter_container ;; gui) gui_container ;;
    stop) stop_container ;; status) status_container ;; *) show_help ;;
esac
