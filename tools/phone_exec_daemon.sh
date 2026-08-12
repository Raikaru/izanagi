#!/data/data/com.termux/files/usr/bin/bash
# phone_exec_daemon.sh — Termux-side executor for the Android phone CI runner.
#
# Why this exists: the GitHub runner lives inside a proot Debian rootfs (the
# runner binary needs glibc). Bionic Vulkan does NOT work under proot — the
# Android loader falls back to "current namespace instead of sphal namespace",
# adrenotools' hook never engages, and vkEnumeratePhysicalDevices returns 0
# devices. Running the same binaries as the Termux uid *outside* proot works.
#
# This daemon watches a spool directory that lives inside the Debian rootfs
# (so both worlds see it with no extra binds) and runs requested scripts
# outside proot, returning stdout/stderr and the exit code.
#
# Trust note: proot's "root" is the same Android uid as Termux, so this grants
# the workflow no privileges it did not already have — but it DOES let workflow
# code run outside the proot view. Keep the phone workflow reviewed-main-only.
#
# Start (from a Termux session):   bash tools/phone_exec_daemon.sh &
# Or over adb:
#   adb shell run-as com.termux sh -c '... nohup bash <path>/phone_exec_daemon.sh &'
set -uo pipefail

ROOTFS="${IZ_ROOTFS:-/data/data/com.termux/files/usr/var/lib/proot-distro/containers/debian/rootfs}"
SPOOL="$ROOTFS/var/spool/izanagi-exec"
MAX_SECONDS="${IZ_EXEC_MAX_SECONDS:-2400}"

mkdir -p "$SPOOL" || exit 1
echo "phone-exec: watching $SPOOL (rootfs $ROOTFS)"

while true; do
    date +%s > "$SPOOL/daemon.alive.tmp" 2>/dev/null && mv -f "$SPOOL/daemon.alive.tmp" "$SPOOL/daemon.alive"
    for req in "$SPOOL"/*.cmd; do
        [ -e "$req" ] || continue
        id="$(basename "$req" .cmd)"
        run="$SPOOL/$id.running"
        mv "$req" "$run" 2>/dev/null || continue
        echo "phone-exec: running $id"
        (
            cd "$ROOTFS" || exit 1
            IZ_ROOTFS="$ROOTFS" \
            PATH="/data/data/com.termux/files/usr/bin:/system/bin" \
            timeout "$MAX_SECONDS" bash "$run"
        ) > "$SPOOL/$id.out" 2>&1
        code=$?
        echo "$code" > "$SPOOL/$id.exit.tmp"
        mv -f "$SPOOL/$id.exit.tmp" "$SPOOL/$id.exit"
        rm -f "$run"
        echo "phone-exec: $id finished with $code"
    done
    sleep 1
done
