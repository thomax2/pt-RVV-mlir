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
#include <linux/if.h>
#include <errno.h>
#include <signal.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

#define LED_DEV_PATH    "/dev/demo_dev0"
#define UART_DEV_PATH   "/dev/ttymxc0"
#define DMABUF_EXP_PATH "/dev/dmabuf_exporter"

// --- DMA BUF 相关的 IOCTL ---
#define DMABUF_IOCTL_ALLOC _IOWR('D', 1, int)
#define DMABUF_IOCTL_GET_PHY _IOWR('D', 2, unsigned int)
#define DMABUF_SIZE (2048 * 512) // 2048 字节每包 * 512 个描述符

// --- FEC 网卡驱动零拷贝相关的结构体和 IOCTL ---
struct fec_zc_config {
    unsigned int dma_addr;
    unsigned int size;
    int enable;
};
#define FEC_IOC_SETUP_ZC   (SIOCDEVPRIVATE + 1)
#define FEC_IOC_WAIT_ZC_DATA (SIOCDEVPRIVATE + 2)

#ifndef DRM_CAP_ATOMIC
#define DRM_CAP_ATOMIC 0x9
#endif

volatile sig_atomic_t keep_running = 1;

// 全局共享资源：时间戳与屏幕状态
atomic_long last_rx_timestamp = 0;
int screen_is_on = 1; // 仅由看门狗线程读写，不需要原子锁

// 外部函数：MLIR 模型推理
extern void process_lidar(float *input_payload, float *drm_framebuffer);

void sigint_handler(int dummy) {
    printf("\n\n[警告] 捕获到 Ctrl+C (SIGINT) 信号！正在准备安全退出...\n");
    keep_running = 0;
}

// ---------------------------------------------------------
// sysfs 电源管理（回退方案）
// ---------------------------------------------------------
int set_display_power_sysfs(int state) {
    // 0: FB_BLANK_UNBLANK (点亮), 4: FB_BLANK_POWERDOWN (关闭)
    int fd = open("/sys/class/graphics/fb0/blank", O_WRONLY);
    if (fd < 0) return -1;
    char buf[2];
    snprintf(buf, sizeof(buf), "%d", state);
    write(fd, buf, 1);
    close(fd);
    return 0;
}

// ---------------------------------------------------------
// DRM atomic 电源管理
// ---------------------------------------------------------
void drm_atomic_set_power(int drm_fd, uint32_t crtc_id, uint32_t conn_id,
                          uint32_t crtc_active_prop, uint32_t conn_dpms_prop,
                          int state) {
    drmModeAtomicReq *req = drmModeAtomicAlloc();
    if (!req) return;

    drmModeAtomicAddProperty(req, crtc_id, crtc_active_prop, state ? 1 : 0);
    drmModeAtomicAddProperty(req, conn_id, conn_dpms_prop,
                             state ? DRM_MODE_DPMS_ON : DRM_MODE_DPMS_OFF);

    uint32_t flags = DRM_MODE_ATOMIC_ALLOW_MODESET;
    int ret = drmModeAtomicCommit(drm_fd, req, flags, NULL);
    if (ret < 0) {
        fprintf(stderr, "Atomic commit failed: %s\n", strerror(errno));
    }
    drmModeAtomicFree(req);
}

// ---------------------------------------------------------
// 看门狗线程参数结构体
// ---------------------------------------------------------
struct pm_args {
    int drm_fd;
    uint32_t crtc_id;
    uint32_t conn_id;
    uint32_t crtc_active_prop;
    uint32_t conn_dpms_prop;
    int use_atomic;          // 是否使用 atomic 方法
};

// ---------------------------------------------------------
// 独立线程：电源管理“看门狗”
// ---------------------------------------------------------
void* power_manager_thread(void* arg) {
    struct pm_args *args = (struct pm_args*)arg;

    printf(">> [看门狗] 电源管理线程已启动！(5秒超时机制)\n");
    while (keep_running) {
        long current_time = time(NULL);
        long idle_time = current_time - atomic_load(&last_rx_timestamp);

        if (screen_is_on && idle_time >= 5) {
            printf("\n[看门狗] 5 秒无数据，执行息屏\n");
            if (args->use_atomic) {
                drm_atomic_set_power(args->drm_fd, args->crtc_id, args->conn_id,
                                     args->crtc_active_prop, args->conn_dpms_prop, 0);
            } else {
                set_display_power_sysfs(4); // 熄屏
            }
            screen_is_on = 0;
        } else if (!screen_is_on && idle_time < 5) {
            printf("\n[看门狗] 数据恢复，唤醒屏幕\n");
            if (args->use_atomic) {
                drm_atomic_set_power(args->drm_fd, args->crtc_id, args->conn_id,
                                     args->crtc_active_prop, args->conn_dpms_prop, 1);
            } else {
                set_display_power_sysfs(0); // 亮屏
            }
            screen_is_on = 1;
        }
        sleep(1);
    }
    printf(">> [看门狗] 线程退出。\n");
    return NULL;
}

// ---------------------------------------------------------
// DRM 显存分配与初始化函数
// ---------------------------------------------------------
int setup_drm_display(float **mapped_ptr, uint32_t *fb_id, uint32_t *size_out,
                      uint32_t *crtc_id_out, uint32_t *conn_id_out,
                      uint32_t *crtc_active_prop, uint32_t *conn_dpms_prop) {
    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;

    drmModeRes *res = drmModeGetResources(fd);
    if (!res) { close(fd); return -1; }

    // 1. 找到第一个 connected 的 connector
    drmModeConnector *conn = NULL;
    for (int i = 0; i < res->count_connectors; i++) {
        conn = drmModeGetConnector(fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) break;
        drmModeFreeConnector(conn);
        conn = NULL;
    }
    if (!conn) { drmModeFreeResources(res); close(fd); return -1; }

    // 保存 connector ID
    *conn_id_out = conn->connector_id;

    // 选择第一个模式
    drmModeModeInfo mode = conn->modes[0];

    // 2. 找到可用的 CRTC
    uint32_t crtc_id = 0;
    // 尝试使用 connector 当前绑定的 encoder 的 crtc
    drmModeEncoder *enc = drmModeGetEncoder(fd, conn->encoder_id);
    if (enc) {
        crtc_id = enc->crtc_id;
        drmModeFreeEncoder(enc);
    }
    // 如果没找到，遍历所有可能的 encoder
    if (crtc_id == 0) {
        for (int i = 0; i < conn->count_encoders; i++) {
            enc = drmModeGetEncoder(fd, conn->encoders[i]);
            if (enc) {
                for (int j = 0; j < res->count_crtcs; j++) {
                    if (enc->possible_crtcs & (1 << j)) {
                        crtc_id = res->crtcs[j];
                        break;
                    }
                }
                drmModeFreeEncoder(enc);
                if (crtc_id) break;
            }
        }
    }
    if (crtc_id == 0) {
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close(fd);
        return -1;
    }
    *crtc_id_out = crtc_id;

    // 3. 创建 dumb buffer
    struct drm_mode_create_dumb create_req = {};
    create_req.width = mode.hdisplay;
    create_req.height = mode.vdisplay;
    create_req.bpp = 32;
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
        perror("CREATE_DUMB failed");
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close(fd);
        return -1;
    }

    // 4. 添加 framebuffer
    if (drmModeAddFB(fd, mode.hdisplay, mode.vdisplay, 24, 32,
                     create_req.pitch, create_req.handle, fb_id) < 0) {
        perror("drmModeAddFB failed");
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close(fd);
        return -1;
    }

    // 5. 映射到用户空间
    struct drm_mode_map_dumb map_req = {};
    map_req.handle = create_req.handle;
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
        perror("MAP_DUMB failed");
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close(fd);
        return -1;
    }
    *mapped_ptr = (float *)mmap(NULL, create_req.size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, map_req.offset);
    if (*mapped_ptr == MAP_FAILED) {
        perror("mmap failed");
        drmModeFreeConnector(conn);
        drmModeFreeResources(res);
        close(fd);
        return -1;
    }
    *size_out = create_req.size;

    // 6. 设置 CRTC，显示 framebuffer
    if (drmModeSetCrtc(fd, crtc_id, *fb_id, 0, 0, &conn->connector_id, 1, &mode) < 0) {
        perror("drmModeSetCrtc failed");
        // 继续执行，但屏幕可能不亮
    }

    // 7. 获取属性 ID（此时 crtc 和 connector 已激活）
    // 注意：conn 结构体即将被释放，但属性获取需要使用 conn->connector_id，我们已经保存到 *conn_id_out
    // 获取 CRTC 的 ACTIVE 属性
    drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(fd, crtc_id, DRM_MODE_OBJECT_CRTC);
    if (props) {
        for (uint32_t i = 0; i < props->count_props; i++) {
            drmModePropertyPtr p = drmModeGetProperty(fd, props->props[i]);
            if (p && (p->flags & DRM_MODE_PROP_ATOMIC)) {
                if (strcmp(p->name, "ACTIVE") == 0) {
                    *crtc_active_prop = p->prop_id;
                }
            }
            if (p) drmModeFreeProperty(p);
        }
        drmModeFreeObjectProperties(props);
    }

    // 获取 Connector 的 DPMS 属性
    props = drmModeObjectGetProperties(fd, *conn_id_out, DRM_MODE_OBJECT_CONNECTOR);
    if (props) {
        for (uint32_t i = 0; i < props->count_props; i++) {
            drmModePropertyPtr p = drmModeGetProperty(fd, props->props[i]);
            if (p && strcmp(p->name, "DPMS") == 0) {
                *conn_dpms_prop = p->prop_id;
                drmModeFreeProperty(p);
                break;
            }
            if (p) drmModeFreeProperty(p);
        }
        drmModeFreeObjectProperties(props);
    }

    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    printf(">> DRM 初始化成功: %dx%d, 显存 %llu 字节, CRTC ID %u\n",
           mode.hdisplay, mode.vdisplay, (unsigned long long)create_req.size, crtc_id);
    return fd;
}

// ---------------------------------------------------------
// 串口初始化
// ---------------------------------------------------------
int uart_setup(const char *device) {
    int fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("Open UART failed");
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag |= (CLOCAL | CREAD);

    tcsetattr(fd, TCSANOW, &options);
    return fd;
}

// ---------------------------------------------------------
// LED 触发
// ---------------------------------------------------------
void trigger_led(int fd_led) {
    char val = '0';
    write(fd_led, &val, 1);
    usleep(100000);
    val = '1';
    write(fd_led, &val, 1);
}

// ---------------------------------------------------------
// 主函数
// ---------------------------------------------------------
int main(int argc, char **argv) {
    int uart_fd = -1, led_fd = -1, exp_fd = -1, drm_fd = -1;
    int dma_buf_fd = -1;
    char *shared_mem = NULL;
    unsigned int phy_addr = 0;
    uint32_t drm_fb_id = 0;
    float *drm_ptr = NULL;
    uint32_t drm_size = 0;
    uint32_t drm_crtc_id = 0, drm_conn_id = 0;
    uint32_t drm_crtc_active_prop = 0, drm_conn_dpms_prop = 0;

    signal(SIGINT, sigint_handler);

    // 0. 初始化 DRM
    drm_fd = setup_drm_display(&drm_ptr, &drm_fb_id, &drm_size,
                               &drm_crtc_id, &drm_conn_id,
                               &drm_crtc_active_prop, &drm_conn_dpms_prop);
    int use_atomic = 0;
    if (drm_fd < 0 || drm_ptr == NULL) {
        printf("[降级] DRM 初始化失败，使用普通 malloc 内存测试 MLIR。\n");
        drm_size = 480 * 272 * 4;
        drm_ptr = (float *)malloc(drm_size);
        if (!drm_ptr) {
            perror("malloc failed");
            return -1;
        }
    } else {
        // 检查驱动是否支持 atomic
        uint64_t has_atomic = 0;
        if (drmGetCap(drm_fd, DRM_CAP_ATOMIC, &has_atomic) == 0 && has_atomic) {
            printf("DRM 驱动支持 Atomic，将使用 atomic 电源管理。\n");
            use_atomic = 1;
        } else {
            printf("DRM 驱动不支持 Atomic，将使用 sysfs 回退方案。\n");
        }
    }
    memset(drm_ptr, 0, drm_size);

    // 启动时强制亮屏并设置初始时间
    if (use_atomic) {
        drm_atomic_set_power(drm_fd, drm_crtc_id, drm_conn_id,
                             drm_crtc_active_prop, drm_conn_dpms_prop, 1);
    } else {
        set_display_power_sysfs(0);
    }
    atomic_store(&last_rx_timestamp, time(NULL));

    // 启动看门狗线程
    pthread_t pm_tid;
    struct pm_args pm_arg = {
        .drm_fd = drm_fd,
        .crtc_id = drm_crtc_id,
        .conn_id = drm_conn_id,
        .crtc_active_prop = drm_crtc_active_prop,
        .conn_dpms_prop = drm_conn_dpms_prop,
        .use_atomic = use_atomic
    };
    if (pthread_create(&pm_tid, NULL, power_manager_thread, &pm_arg) != 0) {
        perror("创建电源管理线程失败");
    }

    // 1. 打开 LED 和 UART
    led_fd = open(LED_DEV_PATH, O_RDWR);
    if (led_fd < 0) perror("open LED failed");
    uart_fd = uart_setup(UART_DEV_PATH);
    if (uart_fd < 0) perror("uart_setup failed");

    // 2. 申请 DMA-BUF
    exp_fd = open(DMABUF_EXP_PATH, O_RDWR);
    if (exp_fd < 0) {
        perror("open dmabuf_exporter failed");
        goto cleanup;
    }
    if (ioctl(exp_fd, DMABUF_IOCTL_ALLOC, &dma_buf_fd) < 0) {
        perror("DMABUF_IOCTL_ALLOC failed");
        goto cleanup;
    }
    if (ioctl(exp_fd, DMABUF_IOCTL_GET_PHY, &phy_addr) < 0) {
        printf("无法获取 DMA-BUF 物理地址！驱动劫持失败。\n");
        goto cleanup;
    }
    printf("分配到的物理地址: 0x%08x\n", phy_addr);
    shared_mem = mmap(NULL, DMABUF_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, dma_buf_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("mmap dma_buf failed");
        goto cleanup;
    }
    memset(shared_mem, 0, DMABUF_SIZE);

    // 3. 配置网卡零拷贝
    int dummy_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (dummy_sock < 0) {
        perror("socket failed");
        goto cleanup;
    }
    struct ifreq ifr;
    strncpy(ifr.ifr_name, "eth0", IFNAMSIZ);
    struct fec_zc_config zc_cfg = {
        .dma_addr = phy_addr,
        .size = DMABUF_SIZE,
        .enable = 1
    };
    ifr.ifr_data = (caddr_t)&zc_cfg;
    if (ioctl(dummy_sock, FEC_IOC_SETUP_ZC, &ifr) < 0) {
        perror("配置网卡零拷贝失败");
        close(dummy_sock);
        goto cleanup;
    }
    close(dummy_sock);
    printf("成功劫持 eth0 接收队列！\n");
    system("ifconfig eth0 up");

    // 4. 主循环
    printf("开始阻塞等待网络数据... (CPU 占用 0%%)\n");
    int ring_idx = 0;
    while (keep_running) {
        // 创建新的 socket 用于 ioctl 等待（每次都要新 socket 吗？可以重用，但简单起见每次新建）
        int wait_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (wait_sock < 0) {
            perror("socket wait");
            break;
        }
        if (ioctl(wait_sock, FEC_IOC_WAIT_ZC_DATA, &ifr) < 0) {
            if (errno == EINTR) {
                close(wait_sock);
                continue;
            }
            perror("等待网卡数据失败");
            close(wait_sock);
            break;
        }
        close(wait_sock);

        volatile unsigned char *current_pkt = (unsigned char *)(shared_mem + (ring_idx * 2048));

        if (current_pkt[0] != 0 || current_pkt[1] != 0) {
            unsigned char *eth_frame = (unsigned char *)current_pkt + 2; // 跳过硬件头

            // 检查 EtherType = IPv4
            if (eth_frame[12] == 0x08 && eth_frame[13] == 0x00) {
                unsigned char ihl = eth_frame[14] & 0x0F;
                unsigned int ip_header_len = ihl * 4;

                // 检查协议 = UDP
                if (eth_frame[14 + 9] == 0x11) {
                    unsigned char *udp_header = eth_frame + 14 + ip_header_len;
                    unsigned short dest_port = (udp_header[2] << 8) | udp_header[3];

                    if (dest_port == 8888) {
                        unsigned short udp_len = (udp_header[4] << 8) | udp_header[5];
                        unsigned short payload_len = udp_len - 8;
                        char *payload = (char *)(udp_header + 8);

                        if (payload_len >= 256) {
                            atomic_store(&last_rx_timestamp, time(NULL));
                            float *tensor_in = (float *)payload;

                            printf("\n[网络接收] 成功捕获 64 个浮点数！\n");
                            printf("输入特征采样: [%.2f, %.2f, %.2f, %.2f...]\n",
                                   tensor_in[0], tensor_in[1], tensor_in[2], tensor_in[3]);

                            process_lidar(tensor_in, drm_ptr);

                            printf("[MLIR 运算完成 (Ring %d)] 神经网络输出结果:\n", ring_idx);
                            for (int i = 0; i < 5; i++) {
                                printf("  Out[%d] = %f\n", i, drm_ptr[i]);
                            }

                            if (led_fd > 0) trigger_led(led_fd);
                        } else {
                            printf("[警告] 收到的 UDP 包长度(%d)不足 256 字节，忽略。\n", payload_len);
                        }
                    }
                }
            }
            memset((void *)current_pkt, 0, 2048);
        }
        ring_idx = (ring_idx + 1) % 512;
    }

cleanup:
    printf("\n执行资源清理工作...\n");

    // 等待看门狗线程退出
    pthread_join(pm_tid, NULL);

    // 关闭网卡
    system("ifconfig eth0 down");
    printf("- 网卡 eth0 已关闭。\n");

    if (shared_mem && shared_mem != MAP_FAILED)
        munmap(shared_mem, DMABUF_SIZE);
    if (dma_buf_fd >= 0) close(dma_buf_fd);
    if (exp_fd >= 0) close(exp_fd);
    if (led_fd >= 0) close(led_fd);
    if (uart_fd >= 0) close(uart_fd);

    // 退出前强制亮屏
    if (use_atomic) {
        drm_atomic_set_power(drm_fd, drm_crtc_id, drm_conn_id,
                             drm_crtc_active_prop, drm_conn_dpms_prop, 1);
    } else {
        set_display_power_sysfs(0);
    }

    if (drm_fd >= 0) {
        // 解除显存映射
        if (drm_ptr && drm_ptr != MAP_FAILED)
            munmap(drm_ptr, drm_size);
        close(drm_fd);
    } else if (drm_ptr) {
        free(drm_ptr);
    }

    printf("- 所有资源已释放，安全退出。\n");
    return 0;
}