#!/usr/bin/env python3
"""VR遥操作IK演示 - 极简版

使用新的简化API，代码量最少的VR遥操作示例。

使用方法:
  python demo_ik_vr.py              # 仅仿真
  python demo_ik_vr.py --sim2real   # 连接真机
"""

import sys
import os
import signal
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from taks.ik import IKController
from taks.vr import VRController


def main():
    parser = argparse.ArgumentParser(description="VR遥操作IK演示")
    parser.add_argument("--sim2real", action="store_true", help="连接真机")
    parser.add_argument("--host", type=str, default="192.168.5.12", help="真机IP")
    parser.add_argument("--freq", type=float, default=50.0, help="控制频率(Hz)")
    args = parser.parse_args()

    # 初始化IK和VR
    ik = IKController()
    vr = VRController()
    vr.start()
    rate = RateLimiter(frequency=args.freq, warn=False)
    ik.dt = rate.dt

    # 真机连接（可选）
    client = None
    if args.sim2real:
        import taks

        taks.connect(args.host, wait_data=True)
        client = taks.register("Semi-Taks-T1")
        print(f"[真机] 已连接 {args.host}")

    # 信号处理
    running = [True]

    def sig_handler(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("=" * 50)
    print("  VR遥操作IK演示 - 极简版")
    print("=" * 50)
    print("  VR手柄: A=复位  B=解冻/复位")
    print("  键盘: Backspace=复位")
    print("=" * 50)

    def key_callback(keycode):
        if keycode == 259:  # Backspace
            if ik.is_frozen:
                ik.unfreeze()
                vr.reset_offset()
            else:
                ik.reset()
                vr.reset_offset()

    # 主循环
    with mujoco.viewer.launch_passive(ik.model, ik.data, show_left_ui=False, show_right_ui=False, key_callback=key_callback) as viewer:
        mujoco.mjv_defaultFreeCamera(ik.model, viewer.cam)
        frame_count = 0

        while viewer.is_running() and running[0]:
            # 核心逻辑：仅需2行
            targets = vr.step(ik)  # VR数据→末端目标
            cmd = ik.step(**targets)  # 末端目标→MIT命令

            # 发送到真机
            if client:
                client.controlMIT(cmd)

            # 渲染
            mujoco.mj_camlight(ik.model, ik.data)
            viewer.sync()

            # 打印状态
            frame_count += 1
            if frame_count % 50 == 0:
                left_pos = ik.left_hand.pos()
                right_pos = ik.right_hand.pos()
                left_grip, right_grip = vr.gripper
                print(
                    f"[L] pos=[{left_pos[0]:.3f},{left_pos[1]:.3f},{left_pos[2]:.3f}] grip={left_grip:.2f} | "
                    f"[R] pos=[{right_pos[0]:.3f},{right_pos[1]:.3f},{right_pos[2]:.3f}] grip={right_grip:.2f}"
                )

            rate.sleep()

    # 清理
    vr.close()
    if client:
        import taks

        taks.disconnect()
    print("[完成] 程序退出")


if __name__ == "__main__":
    main()
