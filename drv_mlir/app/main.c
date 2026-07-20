#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <termios.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <errno.h>

// 配置部分：根据你的板子实际情况修改
#define LED_DEV_PATH    "/dev/demo_dev0"    // 你的驱动生成的设备节点名
#define UART_DEV_PATH   "/dev/ttymxc0"      // 你的板子连接上位机的串口号 (可能是 ttyS0, ttySAC0 等)
#define DMABUF_EXP_PATH "/dev/dmabuf_exporter" // dmabuf driver
#define UDP_PORT        8888                // UDP监听端口

// IOCTL 定义 (必须与驱动一致)
#define DMABUF_IOCTL_ALLOC _IOWR('D', 1, int)
#define DMABUF_SIZE 4096

// 初始化串口
int uart_setup(const char *device) {
    int fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("Open UART failed");
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    // 设置波特率 115200 (根据需要修改)
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    // 8N1 模式 (8数据位, 无校验, 1停止位)
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    
    // 启用接收
    options.c_cflag |= (CLOCAL | CREAD);

    tcsetattr(fd, TCSANOW, &options);
    return fd;
}

// 控制LED
void trigger_led(int fd_led) {
    char val = '0';
    write(fd_led, &val, 1); // 亮
    usleep(100000);         // 延时 100ms
    val = '1';
    write(fd_led, &val, 1); // 灭
}

int main(int argc, char **argv) {
    int sockfd, uart_fd, led_fd, exp_fd;
    int dma_buf_fd = -1;
    char *shared_mem = NULL;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);
    char result_buf[128];
    int a, b, sum, n;

    // 1. 打开 LED 驱动设备
    led_fd = open(LED_DEV_PATH, O_RDWR);
    if (led_fd < 0) {
        printf("无法打开LED驱动: %s (请确认驱动已加载)\n", LED_DEV_PATH);
        // 为了演示，这里不退出，继续跑网络部分
    }

    // 2. 初始化串口
    uart_fd = uart_setup(UART_DEV_PATH);
    if (uart_fd < 0) {
        printf("无法打开串口: %s\n", UART_DEV_PATH);
        return -1;
    }

    // 3. open dma-buf
    exp_fd = open(DMABUF_EXP_PATH, O_RDWR);
    if (exp_fd < 0) {
        perror("无法打开 DMABUF 驱动");
        return -1;
    }
    // --- 2. 申请 DMA-BUF ---
    // 通过 IOCTL 让驱动分配内存，并拿回 fd
    if (ioctl(exp_fd, DMABUF_IOCTL_ALLOC, &dma_buf_fd) < 0) {
        perror("IOCTL Alloc failed");
        return -1;
    }
    printf("成功获取 DMA-BUF FD: %d\n", dma_buf_fd);

    // --- 3. 映射到用户空间 (User Space Mapping) ---
    // 这一步之后，shared_mem 直接指向了内核的那一页物理内存
    shared_mem = mmap(NULL, DMABUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, dma_buf_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }
    printf("DMA-BUF 映射地址: %p\n", shared_mem);

    // 3. 创建 UDP Socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY; // 监听所有网卡
    server_addr.sin_port = htons(UDP_PORT);

    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        return -1;
    }

    printf("应用启动成功！\n");
    printf("UDP监听端口: %d\n", UDP_PORT);
    printf("结果将发送至: %s\n", UART_DEV_PATH);

    // 4. 主循环
    while (1) {
        // 接收 UDP 数据
        // 假设发送的数据格式是纯文本，例如: "10 20"
        n = recvfrom(sockfd, shared_mem, DMABUF_SIZE - 1, MSG_WAITALL, 
                        (struct sockaddr *)&client_addr, &addr_len);

        // 收到数据，让LED闪一下
        if (led_fd > 0) {
            trigger_led(led_fd);
        }

        if(n > 0) {
            // 此时 shared_mem 里的数据已经是内核那块内存里的数据了
            // 如果你有其他驱动（比如 GPU/VPU）拿到 dma_buf_fd，它们就能直接处理这块数据

            printf("[DMA Recv] %s\n", shared_mem);
            trigger_led(led_fd);

            // 解析计算 (仍然在 CPU 上做)
            if (sscanf(shared_mem, "%d %d", &a, &b) == 2) {
                sum = a + b;
                sprintf(result_buf, "DMA_CALC: %d + %d = %d\r\n", a, b, sum);
                write(uart_fd, result_buf, strlen(result_buf));
            }
        }
    }

    munmap(shared_mem, DMABUF_SIZE);
    close(dma_buf_fd);
    close(exp_fd);
    close(sockfd);
    close(uart_fd);
    if (led_fd > 0) close(led_fd);
    return 0;
}