#!/usr/bin/env bash
# phone_exec.sh — run a script OUTSIDE proot, from inside the proot runner.
#
# Reads the script body on stdin, hands it to phone_exec_daemon.sh (running as
# the Termux uid), streams back stdout/stderr and propagates the exit code.
# The script runs with $IZ_ROOTFS set to the Debian rootfs path as seen from
# Android, so guest paths are addressed as "$IZ_ROOTFS/<guest path>".
#
#   echo 'uname -a' | tools/phone_exec.sh
set -uo pipefail

SPOOL=/var/spool/izanagi-exec
TIMEOUT="${IZ_EXEC_TIMEOUT:-2400}"
mkdir -p "$SPOOL"

alive="$SPOOL/daemon.alive"
if [ ! -f "$alive" ] || [ $(( $(date +%s) - $(cat "$alive" 2>/dev/null || echo 0) )) -gt 60 ]; then
    echo "phone-exec: daemon not running (no fresh heartbeat at $alive)." >&2
    echo "            Start it in Termux: bash tools/phone_exec_daemon.sh &" >&2
    exit 127
fi

id="job-$$-$(date +%s%N)"
cat > "$SPOOL/$id.cmd.tmp"
mv "$SPOOL/$id.cmd.tmp" "$SPOOL/$id.cmd"

waited=0
while [ ! -f "$SPOOL/$id.exit" ]; do
    if [ "$waited" -ge "$TIMEOUT" ]; then
        echo "phone-exec: timed out after ${TIMEOUT}s waiting for $id" >&2
        rm -f "$SPOOL/$id.cmd" "$SPOOL/$id.running"
        exit 124
    fi
    sleep 1
    waited=$((waited + 1))
done

cat "$SPOOL/$id.out"
code="$(cat "$SPOOL/$id.exit")"
rm -f "$SPOOL/$id.out" "$SPOOL/$id.exit"
exit "$code"
