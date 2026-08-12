#!/usr/bin/env bash
# run_hardware_qualification.sh — hardware-light qualification bundle for Izanagi.
#
# Probe-first, non-authoritative: the capability report decides whether the
# compiled profile can run on the selected Vulkan device under the SAME
# forced configuration being qualified. If it reports profile_supported:
# false (a legitimate SKIP), the evidence is archived and the script exits
# 0. If the probe itself fails (missing ICD, tool crash, no JSON) the
# script exits 1 — infrastructure failures are never masked as "unsupported".
# A qualified device runs the full API suite; capability JSON + suite log are
# archived for the hardware-qualification issue template.
#
# Usage:
#   ./tools/run_hardware_qualification.sh [--profile native|bindless]
#                                         [--build DIR] [--out DIR]
#                                         [--force-static] [--force-legacy-copy]
#
# The default profile follows the build tree's IZANAGI_VK_PROFILE; --build
# defaults to ./build. Environment variables (IZANAGI_FORCE_*) are passed
# through unchanged — this script never fabricates results.
set -euo pipefail

PROFILE=""
BUILD_DIR=""
OUT_DIR=""
FORCE_STATIC=0
FORCE_LEGACY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --build) BUILD_DIR="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --force-static) FORCE_STATIC=1; shift ;;
        --force-legacy-copy) FORCE_LEGACY=1; shift ;;
        --help|-h)
            sed -n '2,15p' "$0"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
if [[ ! -x "$BUILD_DIR/bin/izanagi_capability_report" ]]; then
    echo "ERROR: $BUILD_DIR/bin/izanagi_capability_report missing — configure + build first." >&2
    exit 2
fi

# The forced configuration applies to BOTH the probe and the suite, so the
# archived capability JSON describes exactly what was qualified.
FORCE_ENV=()
if [[ "$FORCE_STATIC" == 1 ]]; then FORCE_ENV+=(IZANAGI_FORCE_STATIC_GRAPHICS_STATE=1); fi
if [[ "$FORCE_LEGACY" == 1 ]]; then FORCE_ENV+=(IZANAGI_FORCE_LEGACY_COPY_COMMANDS=1); fi

OUT_DIR="${OUT_DIR:-$ROOT/qualification-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

echo "==> Capability probe (profile: ${PROFILE:-auto}, forces: ${FORCE_ENV[*]:-none})"
set +e
env "${FORCE_ENV[@]}" "$BUILD_DIR/bin/izanagi_capability_report" \
    > "$OUT_DIR/capability.json" 2> "$OUT_DIR/capability.err"
PROBE_EXIT=$?
set -e
cat "$OUT_DIR/capability.json"

if [[ $PROBE_EXIT -ne 0 ]]; then
    if grep -q '"profile_supported": false' "$OUT_DIR/capability.json"; then
        echo "==> SKIP: the profile cannot run on this device (non-authoritative; no suite run)."
        echo "    Evidence archived at $OUT_DIR"
        exit 0
    fi
    echo "ERROR: capability probe failed (exit $PROBE_EXIT) without a profile_supported:false" >&2
    echo "       decision — missing ICD, tool crash, or malformed JSON. See $OUT_DIR/capability.err" >&2
    exit 1
fi

if ! grep -q '"profile_supported": true' "$OUT_DIR/capability.json"; then
    echo "ERROR: probe exited 0 without profile_supported:true — malformed JSON?" >&2
    exit 1
fi

if [[ -z "$PROFILE" ]]; then
    PROFILE="$(sed -n 's/.*"profile": "\([^"]*\)".*/\1/p' "$OUT_DIR/capability.json" | head -1)"
fi
if [[ ! -x "$BUILD_DIR/bin/izanagi_tests" ]]; then
    echo "ERROR: $BUILD_DIR/bin/izanagi_tests missing — the test suite is not built." >&2
    exit 2
fi

echo "==> Running the API suite (profile: ${PROFILE:-unknown}, forces: ${FORCE_ENV[*]:-none})"
if ! env "${FORCE_ENV[@]}" "$BUILD_DIR/bin/izanagi_tests" > "$OUT_DIR/suite.log" 2>&1; then
    echo "ERROR: suite failed — see $OUT_DIR/suite.log" >&2
    exit 1
fi

if grep -q "ALL TESTS PASSED" "$OUT_DIR/suite.log"; then
    echo "==> PASS: full suite green on this device."
else
    echo "ERROR: suite did not report ALL TESTS PASSED — see $OUT_DIR/suite.log" >&2
    exit 1
fi

echo "    Evidence archived at $OUT_DIR (capability.json, suite.log)"
