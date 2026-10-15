#!/bin/bash
# 测速 pod 入口：装 RunPod 注入的公钥、起 sshd，然后常驻（测速脚本经 ssh 驱动）
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[ -n "${PUBLIC_KEY:-}" ] && echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no
# 带上环境变量给 ssh 会话（RunPod 的变量默认只在 PID 1 里）
env | grep -E '^(RUNPOD_|PUBLIC_KEY|CUDA|NVIDIA|PATH|LD_LIBRARY_PATH|HF_)' | sed 's/^/export /' > /etc/profile.d/runpod_env.sh
echo "bench pod ready $(date -u)"
sleep infinity
