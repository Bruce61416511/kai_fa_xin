import subprocess
import sys
import os
import time

PORT = 8080

def find_and_kill(port):
    """查找占用指定端口的进程并杀掉"""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                print(f"端口 {port} 被 PID {pid} 占用，正在终止...")
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                time.sleep(1)
                print(f"已终止 PID {pid}")
                return
        print(f"端口 {port} 未被占用")
    except Exception as e:
        print(f"查找/终止进程出错: {e}")

if __name__ == "__main__":
    find_and_kill(PORT)
    print(f"正在启动开发信生成器 -> http://localhost:{PORT}")
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        proc.terminate()
        proc.wait()
        print("服务已关闭")
