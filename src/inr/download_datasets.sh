#!/usr/bin/env bash
# Download MNIST, FashionMNIST, and CIFAR-10 to the expected torchvision layout.
# yann.lecun.com is unreliable; this script uses PyTorch S3 and official mirrors.
#
# Usage:
#   bash src/inr/download_datasets.sh [DATA_PATH]
#
# DATA_PATH defaults to /nfs/stak/users/hongsa/INR-Robustness/assets/raw

set -eo pipefail

DATA_PATH="${1:-/nfs/stak/users/hongsa/INR-Robustness/assets/raw}"

# ── MNIST ────────────────────────────────────────────────────────────────────
MNIST_DIR="${DATA_PATH}/MNIST/raw"
mkdir -p "${MNIST_DIR}"

echo "=== Downloading MNIST ==="
MNIST_MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
for f in \
    train-images-idx3-ubyte.gz \
    train-labels-idx1-ubyte.gz \
    t10k-images-idx3-ubyte.gz \
    t10k-labels-idx1-ubyte.gz
do
    if [[ ! -f "${MNIST_DIR}/${f%.gz}" ]]; then
        echo "  Downloading ${f}..."
        wget -q --show-progress -P "${MNIST_DIR}" "${MNIST_MIRROR}/${f}"
        gunzip -f "${MNIST_DIR}/${f}"
    else
        echo "  Already exists: ${f%.gz}"
    fi
done

# ── FashionMNIST ─────────────────────────────────────────────────────────────
FASHION_DIR="${DATA_PATH}/FashionMNIST/raw"
mkdir -p "${FASHION_DIR}"

echo "=== Downloading FashionMNIST ==="
FASHION_MIRROR="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
for f in \
    train-images-idx3-ubyte.gz \
    train-labels-idx1-ubyte.gz \
    t10k-images-idx3-ubyte.gz \
    t10k-labels-idx1-ubyte.gz
do
    if [[ ! -f "${FASHION_DIR}/${f%.gz}" ]]; then
        echo "  Downloading ${f}..."
        wget -q --show-progress -P "${FASHION_DIR}" "${FASHION_MIRROR}/${f}"
        gunzip -f "${FASHION_DIR}/${f}"
    else
        echo "  Already exists: ${f%.gz}"
    fi
done

# ── CIFAR-10 ─────────────────────────────────────────────────────────────────
CIFAR_DIR="${DATA_PATH}"
mkdir -p "${CIFAR_DIR}"

echo "=== Downloading CIFAR-10 ==="
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TAR="${CIFAR_DIR}/cifar-10-python.tar.gz"
if [[ ! -d "${CIFAR_DIR}/cifar-10-batches-py" ]]; then
    echo "  Downloading cifar-10-python.tar.gz..."
    wget -q --show-progress -P "${CIFAR_DIR}" "${CIFAR_URL}"
    tar -xzf "${CIFAR_TAR}" -C "${CIFAR_DIR}"
    rm -f "${CIFAR_TAR}"
else
    echo "  Already exists: cifar-10-batches-py"
fi

echo ""
echo "All datasets ready in: ${DATA_PATH}"
