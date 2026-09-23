# Ubuntu 26.04 + ROS 2 Lyrical on the Jetson: container now, native after IROS

Written 2026-09-23. Decision: run Lyrical in a container on the car today; do the native
upgrade only after the RoboRacer race at IROS (Sept 27-30), on a cloned SD card, with a person
who has the `sudo` password at the keyboard.

## Where the car is

| Item | Value |
|---|---|
| Board | Jetson Orin Nano Super Developer Kit, 256 GB SD card (`mmcblk0`), no NVMe |
| BSP | Jetson Linux R39.2.1 = JetPack 7.2.1, kernel 6.8 (tegra) |
| OS | Ubuntu 24.04.4 LTS, Python 3.12.3 |
| ROS | ROS 2 Jazzy (binary), workspaces `~/f1tenth_ws` (23 pkgs) and `~/atlas_ws` |
| GPU stack | CUDA 13.2, TensorRT 10.16.2 (`python3-libnvinfer` pinned to Python >= 3.12, < 3.13) |
| Access | user `atlas` is in `sudo` but has no passwordless sudo; in `docker` group |
| Network | none of its own; reaches the internet only through the laptop (see below) |
| Clock | no RTC battery / NTP source: boots at a stale date (read 2026-09-05 on 9/23) |

## Why not upgrade in place now

1. **No supported BSP.** JetPack 7.2 is built on Ubuntu 24.04 and kernel 6.8. NVIDIA does not
   publish an Ubuntu 26.04 root filesystem for Orin. A release upgrade would put a 26.04 userspace
   on a 24.04 BSP (kernel modules, `nvidia-l4t-*` packages, firmware), which NVIDIA documents as
   unsupported for partially upgraded platform components.
2. **A hard Python conflict.** Ubuntu 26.04 ships Python 3.14. The installed TensorRT Python
   binding requires `python3 < 3.13`, and every compiled Jazzy Python extension (rclpy) targets 3.12.
   The 9/11 dry run (`~/ubuntu26-native-preflight/dist-upgrade.log`) would pull 26.04 packages over
   66 held ones; it cannot resolve cleanly.
3. **No way back.** One root partition, no A/B rootfs slot, no second disk. A failed upgrade four
   days before IROS means no car at IROS.
4. **No sudo** for this account, which is correct: an OS upgrade should be done by a person.

## What exists today (container)

`docker/lyrical/Dockerfile` + `hardware/scripts/lyrical_container.sh` build an Ubuntu 26.04 +
ROS 2 Lyrical image on the car and test this repository inside it:

```
# on the laptop (the car has no internet): a small allowlisted proxy, then a reverse tunnel
python3 tiny_proxy.py 127.0.0.1 &          # see docs/REMOTE.md
ssh -N -R 3128:127.0.0.1:3128 atlas@192.168.55.1 &
# on the car
PROXY=http://127.0.0.1:3128 bash hardware/scripts/lyrical_container.sh build
bash hardware/scripts/lyrical_container.sh test     # colcon build + imports + unit tests
bash hardware/scripts/lyrical_container.sh shell    # /dev, NVIDIA runtime, host network
```

apt inside the build runs with `Acquire::Check-Date=false` because of the stale car clock.

## Native upgrade plan (after IROS)

0. **Fix the clock first.** `sudo timedatectl set-time "<laptop date>"`, then point chrony at the
   laptop (or any reachable NTP server) so TLS and apt stop failing on dates.
1. **Clone the SD card** on another machine (`dd` or Balena), label the original "JetPack 7.2.1 /
   Jazzy / known good", and do everything below on the clone. Rollback = swap cards.
2. **Record the baseline** on the original: `colcon test` in both workspaces, `pytest tests`,
   sensor rates (`/scan` 10 Hz, `/oakd/rgb`, `/oakd/imu`, `/vesc/odom`), TensorRT engine latencies.
3. **Choose the route**, in order of preference:
   a. *Wait for NVIDIA.* If a JetPack built on Ubuntu 26.04 ships for Orin, flash it (SDK Manager
      or the SD image) and skip steps 4-5. Check the JetPack page before starting.
   b. *Userspace upgrade on the clone.* `sudo apt-mark hold 'nvidia-l4t-*'`, set
      `Prompt=lts`, `sudo do-release-upgrade -d`. Expect to rebuild anything that links Python 3.12.
4. **GPU stack on 26.04.** Keep TensorRT/CUDA from the BSP. For Python inference, either run the
   engines from C++ (`trtexec`, or a small C++ node) or keep a Python 3.12 virtual environment
   (`uv python install 3.12`) for TensorRT and use ONNX Runtime (CPU, 1-2 ms for the student)
   from the system Python 3.14.
5. **ROS 2 Lyrical.** Install `ros-lyrical-ros-base` from packages.ros.org (resolute, arm64).
   Rebuild `~/f1tenth_ws` from source (vesc, rplidar_ros, ackermann_mux, f1tenth_stack, joy),
   then `~/atlas_ws`. Replace `source /opt/ros/jazzy/setup.bash` in scripts with
   `source /opt/ros/${ROS_DISTRO}/setup.bash` (the 9/23 commit already does this for the
   scripts that had Humble hard-coded).
6. **Validate** against the step-2 baseline: every test, every sensor rate, udev symlinks under
   `/dev/sensors/`, DepthAI 2.x with the OAK-D Pro, the VESC over `/dev/ttyACM0`, engine
   latencies, then a wheels-up bench run, then a 2-minute floor drive.
7. **Promote or roll back.** Only if every item in step 6 passes does the clone become the car's
   card. Otherwise put the original back.

## Sources

- NVIDIA JetPack (JetPack 7 on Ubuntu 24.04, kernel 6.8): https://developer.nvidia.com/embedded/jetpack
- Jetson Linux R39.2.1 release notes: https://docs.nvidia.com/jetson/archives/r39.2.1/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.1.pdf
- ROS 2 Lyrical binary install (Ubuntu 26.04): https://docs.ros.org/en/lyrical/Installation/Alternatives/Ubuntu-Install-Binary.html
- Ubuntu 26.04 Python: https://packages.ubuntu.com/resolute/python3
