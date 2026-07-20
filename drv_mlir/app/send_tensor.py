import socket
import struct
import time

# 目标开发板的 IP 和端口
TARGET_IP = "192.168.0.120"
TARGET_PORT = 8888

# 1. 准备你的数据：1x1x8x8 = 64 个浮点数
# 这里我们先全部填充为 1.0，完全模拟你之前的 MLIR 常量初始化
tensor_data = [1.0] * 64

# 2. 将这 64 个 Python 浮点数，严格按照 C 语言的 float 内存格式打包
# '<' 代表小端模式，'64f' 代表 64 个单精度浮点数
binary_payload = struct.pack(f'<{len(tensor_data)}f', *tensor_data)

# 3. 创建 UDP Socket 并发送
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"准备发送 {len(tensor_data)} 个浮点数...")
print(f"二进制负载大小: {len(binary_payload)} 字节")

# 循环发送，每秒发一次，就像激光雷达持续产生数据一样
while True:
    sock.sendto(binary_payload, (TARGET_IP, TARGET_PORT))
    print(">> 数据帧已发送!")
    time.sleep(1)