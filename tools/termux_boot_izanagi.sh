#!/data/data/com.termux/files/usr/bin/sh
# termux_boot_izanagi.sh — unattended recovery of the Izanagi phone CI rig.
#
# Install as ~/.termux/boot/00-izanagi.sh (Termux:Boot runs boot scripts in
# lexical order after BOOT_COMPLETED). Brings back, with no interaction:
#   1. a wake lock (Android otherwise suspends the CPU and stalls jobs)
#   2. the phone-exec daemon that runs bionic tests outside proot
#   3. the GitHub Actions runner inside the proot Debian rootfs
#
# Termux:Boot must have been launched once after install, or Android keeps the
# app in the stopped state and never delivers BOOT_COMPLETED.
PREFIX=/data/data/com.termux/files/usr
HOME=/data/data/com.termux/files/home
PATH=$PREFIX/bin:/system/bin
LD_LIBRARY_PATH=$PREFIX/lib
PROOT_TMP_DIR=$PREFIX/tmp
export PREFIX HOME PATH LD_LIBRARY_PATH PROOT_TMP_DIR

LOG="$HOME/izanagi-boot.log"
echo "=== izanagi boot $(date) ===" >> "$LOG"

# 1. Keep the CPU alive across screen-off and Doze.
termux-wake-lock >> "$LOG" 2>&1 || echo "wake-lock failed" >> "$LOG"
ROOTFS=$PREFIX/var/lib/proot-distro/containers/debian/rootfs
SPOOL=$ROOTFS/var/spool/izanagi-exec

# Liveness helper: scan procfs rather than pgrep/kill -0. Same-uid /proc
# entries stay readable in every launch context, while `kill -0` is denied
# under run-as and `pgrep -f` matches its own invoking shell.
runner_running() {
    for proc_dir in /proc/[0-9]*; do
        # cat (not the shell) must open the file, or the shell itself reports
        # "Permission denied" for every process owned by another uid.
        if cat "$proc_dir/cmdline" 2>/dev/null | tr '\0' ' ' | grep -q 'Runner\.Listener'; then
            return 0
        fi
    done
    return 1
}

# 2. Bionic test executor (must run OUTSIDE proot: the Android loader cannot
#    use the sphal namespace under proot, so no Vulkan ICD loads there).
#    Liveness comes from the daemon's own heartbeat, not pgrep: process
#    visibility differs by launch context, and pgrep -f self-matches.
now=$(date +%s)
beat=$(cat "$SPOOL/daemon.alive" 2>/dev/null || echo 0)
if [ "$((now - beat))" -gt 60 ]; then
    nohup "$PREFIX/bin/bash" "$HOME/bin/phone_exec_daemon.sh" \
        >> "$HOME/phone-exec.log" 2>&1 &
    echo "started phone-exec daemon" >> "$LOG"
else
    echo "phone-exec daemon already alive" >> "$LOG"
fi

# 3. GitHub Actions runner, as the unprivileged 'runner' account in Debian.
#    DOTNET_GCHeapHardLimit works around CoreCLR's heap reservation failing
#    under proot (0x8007000E).
if runner_running; then
    echo "actions runner already alive" >> "$LOG"
else
    nohup "$PREFIX/bin/proot-distro" login debian --user runner -- \
        /bin/bash -lc 'cd ~/actions-runner && export DOTNET_GCHeapHardLimit=1C0000000 && exec ./run.sh' \
        >> "$HOME/runner.log" 2>&1 &
    echo "started actions runner (launcher pid $!)" >> "$LOG"
fi
