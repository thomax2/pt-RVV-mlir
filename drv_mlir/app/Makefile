# 交叉编译器名称
CC = arm-linux-gnueabihf-gcc  

# 编译标志：开启最高级别优化(-O3)，禁用地址随机化(-no-pie，利于裸机内存操作)
# 注意：已经去掉了 -static
CFLAGS = -O3 -no-pie 

# 头文件搜索路径
# 这里对应你之前手动安装的 libdrm 所在的 include 路径
INCLUDES = -I/usr/include/libdrm -I/usr/include

# 链接库路径和指定的动态库
# 告诉链接器去哪里找 libdrm.so，并链接它
LDFLAGS = -L/usr/lib/arm-linux-gnueabihf -ldrm -lpthread

# 目标文件名
TARGET = main_fec              

# 源文件列表
SRCS = main_fec.c output_arm.s 

all:
	# 注意链接顺序：库文件变量 $(LDFLAGS) 必须放在最后面！
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)