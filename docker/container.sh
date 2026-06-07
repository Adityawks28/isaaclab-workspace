#!/bin/bash

# Isaac Sim (Docker) launcher for robot-manipulation-lab
# Fully containerized workflow — no local Isaac Sim install needed.
# Image: nvcr.io/nvidia/isaac-sim:4.5.0  (GPU: RTX 4050 Laptop, 6 GB VRAM)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LAB_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"   # repo root = parent of docker/

CONTAINER_NAME="isaac-sim"
IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
CACHE="${HOME}/docker/isaac-sim"              # persistent caches (warm)

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  help     Show this help message"
    echo "  start    Create + start the container (detached, kept alive)"
    echo "  enter    Open a bash shell inside the running container"
    echo "  gui      Launch the Isaac Sim GUI window (X11)"
    echo "  stop     Stop and remove the container"
    echo "  status   Show whether the container is running"
    echo ""
    echo "Your lab code (${LAB_DIR}) is mounted at /root/lab inside the container."
}

setup_x11() {
    if [ -n "$DISPLAY" ]; then
        echo "Setting up X11 forwarding..."
        xhost +local:root >/dev/null 2>&1 || true
    else
        echo "Warning: DISPLAY is not set — GUI will not be available."
    fi
}

start_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' already exists. Starting it..."
        docker start "${CONTAINER_NAME}" >/dev/null
        echo "Started. Use '$0 enter' or '$0 gui'."
        return
    fi

    setup_x11
    mkdir -p "${CACHE}"/{cache/{kit,ov,pip,glcache,computecache},logs,data,documents}

    echo "Creating Isaac Sim container..."
    docker run --name "${CONTAINER_NAME}" -d --entrypoint bash \
        --runtime=nvidia --gpus all \
        -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
        -e "OMNI_KIT_ALLOW_ROOT=1" \
        -e "DISPLAY=${DISPLAY}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "${LAB_DIR}:/root/lab:rw" \
        -v "${CACHE}/cache/kit:/isaac-sim/kit/cache:rw" \
        -v "${CACHE}/cache/ov:/root/.cache/ov:rw" \
        -v "${CACHE}/cache/pip:/root/.cache/pip:rw" \
        -v "${CACHE}/cache/glcache:/root/.cache/mesa_shader_cache:rw" \
        -v "${CACHE}/cache/computecache:/root/.cache/computecache:rw" \
        -v "${CACHE}/logs:/root/.nvidia-omniverse/logs:rw" \
        -v "${CACHE}/data:/root/.local/share/ov/data:rw" \
        -v "${CACHE}/documents:/root/Documents:rw" \
        "${IMAGE}" -c "tail -f /dev/null"

    echo "Container running. Use '$0 enter' for a shell or '$0 gui' for the window."
}

enter_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Error: container is not running. Run '$0 start' first."
        exit 1
    fi
    setup_x11
    docker exec -it "${CONTAINER_NAME}" bash
}

gui_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Error: container is not running. Run '$0 start' first."
        exit 1
    fi
    setup_x11
    echo "Launching Isaac Sim GUI (first launch can take a few minutes)..."
    docker exec -it -e OMNI_KIT_ALLOW_ROOT=1 "${CONTAINER_NAME}" bash -c "cd /isaac-sim && ./isaac-sim.sh --allow-root"
}

stop_container() {
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' does not exist."
        exit 0
    fi
    echo "Warning: this stops and REMOVES the container. Anything written"
    echo "outside /root/lab and the mounted cache dirs will be lost."
    read -p "Continue? [y/N] " -n 1 -r; echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null
        echo "Removed."
    else
        echo "Cancelled."
    fi
}

status_container() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Running."
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopped (exists). Run '$0 start' to resume."
    else
        echo "Not created. Run '$0 start'."
    fi
}

case "$1" in
    help|"")  show_help ;;
    start)    start_container ;;
    enter)    enter_container ;;
    gui)      gui_container ;;
    stop)     stop_container ;;
    status)   status_container ;;
    *)        echo "Error: unknown command '$1'"; show_help; exit 1 ;;
esac
