#include <linux/module.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/errno.h>
#include <linux/of.h>
#include <linux/of_gpio.h>
#include <linux/gpio.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include "led_opr.h"

/* 定义一个结构体来保存每个设备的信息 */
struct my_gpio_desc {
    int gpio;       /* GPIO 编号 */
    char name[32];  /* 设备名称 (来自 label) */
    int index;      /* 在 /dev/demo_dev 中的索引 */
};

/* 假设最多支持 10 个设备 */
static struct my_gpio_desc g_devs[10];
static int g_dev_count = 0;

/* 硬件初始化函数 */
static int board_init_gpio(int which)
{
    /* 实际上在 probe 时已经 request 了，这里可以做 reset 等操作 */
    /* 也可以留空 */
    printk("Board: Init device %s (GPIO %d)\n", g_devs[which].name, g_devs[which].gpio);
    return 0;
}

/* 硬件控制函数 */
static int board_ctl_gpio(int which, char status)
{
    if (which >= g_dev_count) {
        printk("Board: Error index out of range\n");
        return -1;
    }
    
    printk("Board: Set %s (GPIO %d) to %d\n", g_devs[which].name, g_devs[which].gpio, status);
    
    /* 假设 status 1 是开，0 是关。如果是低电平有效，这里可能要取反，或者 DTS 里指定了 active low */
    /* gpio_set_value 会自动处理 DTS 里的 active low 标志 */
    gpio_set_value(g_devs[which].gpio, status);
    return 0;
}

static struct led_operations my_board_opr = {
    .init = board_init_gpio,
    .ctl  = board_ctl_gpio,
};

static int chip_gpio_probe(struct platform_device *pdev)
{
    struct device_node *parent = pdev->dev.of_node;
    struct device_node *child;
    int ret;
    const char *label_str;
    enum of_gpio_flags flag;
    int gpio;
    g_dev_count = 0;

    printk("Board: Start probing dts nodes...\n");

    /* === 核心：遍历父节点下的所有子节点 === */
    for_each_available_child_of_node(parent, child) {
        
        if (g_dev_count >= 10) break;

        /* 1. 读取 label 属性，如果没写 label，就用节点名 */
        if (of_property_read_string(child, "label", &label_str)) {
            label_str = child->name;
        }
        
        /* 复制名字备用 */
        snprintf(g_devs[g_dev_count].name, 32, "%s", label_str);

        /* 2. 读取 gpios 属性 (注意：教程风格通常用 "gpios" 而不是 "led-gpios") */
        /* index 0 表示取该属性里的第一个 GPIO */
        gpio = of_get_named_gpio_flags(child, "gpios", 0, &flag);
        
        if (!gpio_is_valid(gpio)) {
            printk("Board: Node %s has invalid gpio\n", label_str);
            continue;
        }

        /* 3. 申请 GPIO */
        ret = gpio_request(gpio, label_str);
        if (ret) {
            printk("Board: Failed to request gpio %d for %s\n", gpio, label_str);
            continue;
        }

        /* 4. 设置为输出，默认灭 (0) */
        gpio_direction_output(gpio, 1);

        /* 保存信息 */
        g_devs[g_dev_count].gpio = gpio;
        g_devs[g_dev_count].index = g_dev_count; // 简单的按顺序分配索引
        
        /* 简单逻辑：根据 label 打印一下识别结果 */
        if (strstr(label_str, "led")) {
            printk("Board: Found LED at index %d\n", g_dev_count);
        } else if (strstr(label_str, "beep")) {
            printk("Board: Found BEEP at index %d\n", g_dev_count);
        }

        g_dev_count++;
    }

    /* 将扫描到的数量填入结构体，注册给上层 */
    my_board_opr.num = g_dev_count;
    register_led_opr(&my_board_opr);

    return 0;
}

static int chip_gpio_remove(struct platform_device *pdev)
{
    int i;
    unregister_led_opr();
    
    for (i = 0; i < g_dev_count; i++) {
        gpio_free(g_devs[i].gpio);
    }
    return 0;
}

/* 匹配表：名字必须和 DTS 里的 compatible 一样 */
static const struct of_device_id my_match_table[] = {
    { .compatible = "my,multi-gpios" },
    { }
};

static struct platform_driver chip_gpio_driver = {
    .probe      = chip_gpio_probe,
    .remove     = chip_gpio_remove,
    .driver     = {
        .name   = "my_board_gpio",
        .of_match_table = my_match_table,
    },
};

static int __init board_init(void)
{
    return platform_driver_register(&chip_gpio_driver);
}

static void __exit board_exit(void)
{
    platform_driver_unregister(&chip_gpio_driver);
}

module_init(board_init);
module_exit(board_exit);
MODULE_LICENSE("GPL");