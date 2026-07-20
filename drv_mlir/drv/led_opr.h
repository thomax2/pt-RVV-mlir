#ifndef _LED_OPR_H
#define _LED_OPR_H

struct led_operations {
    int num; /* 支持的设备数量 */
    int (*init)(int which); /* 初始化函数 */
    int (*ctl)(int which, char status); /* 控制函数 */
};

/* 供下层驱动调用，注册硬件操作方法 */
void register_led_opr(struct led_operations *opr);
/* 供下层驱动调用，注销 */
void unregister_led_opr(void);

#endif