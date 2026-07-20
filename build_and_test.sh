#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

TEST_MODE="all"

usage()
{
    echo "Usage: $0 [--quick | --slow-only | --all]"
    echo
    echo "  --quick      Exclude tests labelled 'slow'"
    echo "  --slow-only  Run only tests labelled 'slow'"
    echo "  --all        Run all tests (default)"
}

while (( $# > 0 )); do
    case "$1" in
        --quick)
            TEST_MODE="quick"
            ;;
        --slow-only)
            TEST_MODE="slow"
            ;;
        --all)
            TEST_MODE="all"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

cmake \
    -S . \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build "${BUILD_DIR}" \
    --parallel "${BUILD_JOBS}"

CTEST_ARGS=(
    --test-dir "${BUILD_DIR}"
    --output-on-failure
)

case "${TEST_MODE}" in
    quick)
        echo "Running quick tests; excluding tests labelled 'slow'"
        CTEST_ARGS+=(-LE slow)
        ;;
    slow)
        echo "Running only tests labelled 'slow'"
        CTEST_ARGS+=(-L slow)
        ;;
    all)
        echo "Running all tests"
        ;;
esac

ctest "${CTEST_ARGS[@]}"