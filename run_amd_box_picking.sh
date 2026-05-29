#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-genesis-amd:local}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOW_XHOST=0
BUILD_IMAGE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD_IMAGE=1
            shift
            ;;
        --xhost)
            ALLOW_XHOST=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ "${BUILD_IMAGE}" == "1" ]]; then
    docker build -t "${IMAGE}" -f "${REPO_DIR}/docker/Dockerfile.amdgpu" "${REPO_DIR}"
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_ARGS=(-it)
fi

SHOW_VIEWER=0
for arg in "$@"; do
    if [[ "${arg}" == "-v" || "${arg}" == "--vis" ]]; then
        SHOW_VIEWER=1
    fi
done

find_xauthority() {
    local candidate
    if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
        printf '%s\n' "${XAUTHORITY}"
        return 0
    fi
    if [[ -f "${HOME}/.Xauthority" ]]; then
        printf '%s\n' "${HOME}/.Xauthority"
        return 0
    fi
    for candidate in "/run/user/$(id -u)"/.mutter-Xwaylandauth.*; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
    DISPLAY=:0
fi

XAUTHORITY_PATH=""
if resolved_xauthority="$(find_xauthority)"; then
    XAUTHORITY_PATH="${resolved_xauthority}"
fi

if [[ "${SHOW_VIEWER}" == "1" && -z "${DISPLAY:-}" ]]; then
    cat >&2 <<'EOF'
No display detected for the Genesis viewer.

Run this from a desktop terminal, or export the host X11/XWayland display:
  export DISPLAY=:0

Useful checks:
  echo "$DISPLAY"
  ls -l /tmp/.X11-unix
  ls -l "${XAUTHORITY:-$HOME/.Xauthority}"

If Xauthority is not available, retry with:
  ./run_amd_box_picking.sh --xhost -v

Headless mode still works without a display: omit -v/--vis.
EOF
    exit 2
fi

ENV_ARGS=(-e DISPLAY="${DISPLAY:-}" -e LOCAL_USER_ID="$(id -u)")
if [[ -n "${PYOPENGL_PLATFORM:-}" ]]; then
    ENV_ARGS+=(-e PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}")
elif [[ "${SHOW_VIEWER}" == "1" && -n "${DISPLAY:-}" ]]; then
    ENV_ARGS+=(-e PYOPENGL_PLATFORM=glx)
elif [[ -z "${DISPLAY:-}" ]]; then
    ENV_ARGS+=(-e PYOPENGL_PLATFORM=egl)
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    ENV_ARGS+=(-e HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION}")
fi

DEVICE_GROUP_ARGS=()
add_device_group_arg() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        local gid
        gid="$(stat -c '%g' "${path}")"
        DEVICE_GROUP_ARGS+=("--group-add=${gid}")
    fi
}
add_device_group_arg /dev/kfd
add_device_group_arg /dev/dri/renderD128

VOLUME_ARGS=(-v "${REPO_DIR}:/workspace/genesis-world")
if [[ -n "${DISPLAY:-}" ]]; then
    VOLUME_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    if [[ -n "${XAUTHORITY_PATH}" ]]; then
        ENV_ARGS+=(-e XAUTHORITY=/tmp/.docker.xauth)
        VOLUME_ARGS+=(-v "${XAUTHORITY_PATH}:/tmp/.docker.xauth:ro")
    elif [[ "${ALLOW_XHOST}" == "1" ]]; then
        xhost +SI:localuser:"$(id -un)" >/dev/null
    fi
fi

docker run --rm "${TTY_ARGS[@]}" \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    "${DEVICE_GROUP_ARGS[@]}" \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 8G \
    "${ENV_ARGS[@]}" \
    "${VOLUME_ARGS[@]}" \
    -w /workspace/genesis-world \
    "${IMAGE}" \
    python3 examples/rigid/industrial_box_picking_amd.py "$@"
