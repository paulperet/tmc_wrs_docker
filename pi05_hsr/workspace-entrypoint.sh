#!/bin/bash
# Patched copy of the stock /entrypoint.sh, bind-mounted into the workspace container
# via docker-compose.override.yml. Differences from the original:
#   * shebang drops `-e`            -> a failing setup step can't abort startup
#   * the recursive chown is skipped -> it errors on macOS bind-mounted .git files
#                                       (chown is unsupported there) and that abort
#                                       crash-looped both the container and the
#                                       supervisor-managed `jupyter` program.
# Everything else mirrors the original entrypoint.

USER_ID=$(id -u)
GROUP_ID=$(id -g)

sudo usermod -u $USER_ID -o -m -d /home/developer developer > /dev/null 2>&1 || true
sudo groupmod -g $GROUP_ID developer > /dev/null 2>&1 || true
# sudo chown -R developer:developer /workspace   # skipped: fails on macOS bind mounts

ln -sfn /home/developer/.vscode /workspace/.vscode || true

rm -f /workspace/compile_flags.txt || true
sed -e 's@\$ROS_DISTRO@'"$ROS_DISTRO"'@' /home/developer/compile_flags.txt > /workspace/compile_flags.txt || true

ln -sfn /workspace /home/developer/workspace || true

source /opt/ros/$ROS_DISTRO/setup.bash

mkdir -p /workspace/src && cd /workspace/src && catkin_init_workspace || true

cd /home/developer

exec $@
