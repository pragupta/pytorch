#!/usr/bin/env bash
# Script used only in CD pipeline to build and install rocSHMEM

set -eou pipefail

function do_install() {
    ROCSHMEM_VERSION=e1a7e20b1b4372d38df4bc34872d4edd19b9c7e2
    rocm_dir="/opt/rocm"
    (
        set -x
        curr_dir=$(pwd)
        tmp_dir=$(mktemp -d)

        sudo apt update -y && sudo apt install -y libibverbs-dev
        git clone https://github.com/ROCm/rocSHMEM.git --branch develop
        cd rocSHMEM
        git checkout ${ROCSHMEM_VERSION}

        mkdir build
        cd build
        INSTALL_PREFIX=${rocm_dir} ../scripts/build_configs/all_backends

        cd ${curr_dir}

    )
}

do_install
