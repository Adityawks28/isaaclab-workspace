#!/bin/bash
# One-shot per machine: clone+pin Isaac Lab, then build its Docker image.
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

# 1. Clone Isaac Lab pinned to a 4.5-compatible tag (idempotent).
if [ ! -d "${ISAACLAB_DIR}/.git" ]; then
    echo "Cloning Isaac Lab ${ISAACLAB_TAG}..."
    git clone --depth 1 --branch "${ISAACLAB_TAG}" \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
else
    echo "Isaac Lab already cloned at ${ISAACLAB_DIR}."
fi

# 2. Confirm the Isaac Sim base image is present (pull if missing).
if ! docker image inspect "${ISAACSIM_BASE_IMAGE}" >/dev/null 2>&1; then
    echo "Pulling base image ${ISAACSIM_BASE_IMAGE} (large)..."
    docker pull "${ISAACSIM_BASE_IMAGE}"
fi

# 3. Build + start the Isaac Lab image via its official Docker tooling.
echo "Building Isaac Lab image (first build is slow)..."
python3 "${ISAACLAB_DIR}/docker/container.py" start

# 4. Install Isaac Lab extensions INSIDE the running container.
#    The image build does not fully wire editable installs against the runtime
#    bind-mount, so we (re)install here. Two known v2.1.0 gotchas, handled:
#      a) flatdict (an isaaclab core dep) fails under PEP517 build isolation with
#         "No module named 'pkg_resources'" -> pre-install with --no-build-isolation.
#      b) the installer pulls numpy 2.x, but isaaclab requires numpy<2 -> pin it back.
echo "Installing Isaac Lab extensions inside the container..."
docker exec "${ISAACLAB_CONTAINER}" bash -lc "cd /workspace/isaaclab && \
    ./isaaclab.sh -p -m pip install --no-build-isolation flatdict==4.0.1 && \
    ./isaaclab.sh --install && \
    ./isaaclab.sh -p -m pip install 'numpy==1.26.4'"

echo "Setup complete. Train with: ./scripts/train.sh   (GUI eval: ./scripts/eval.sh)"
