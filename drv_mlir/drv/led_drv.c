#include <linux/module.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/miscdevice.h>
#include <linux/kernel.h>
#include <linux/major.h>
#include <linux/mutex.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/stat.h>
#include <linux/init.h>
#include <linux/device.h>
#include <linux/tty.h>
#include <linux/kmod.h>
#include <linux/gfp.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include "led_opr.h"

#define DEV_NAME "demo_dev"
#define MAX_DEVS 4  /* 最多支持4个设备 */

static struct led_operations *p_led_opr = NULL;
static dev_t devid;
static struct cdev my_cdev;
static struct class *my_class;

/* 打开设备：在这里可以根据次设备号做区分 */
static int led_drv_open(struct inode *node, struct file *file)
{
    int minor = iminor(node);
    if (p_led_opr && p_led_opr->init) {
        return p_led_opr->init(minor);
    }
    return 0;
}

/* 写设备：控制 LED/Beep */
static ssize_t led_drv_write(struct file *file, const char __user *buf, size_t size, loff_t *offset)
{
    int ret;
    char status; // 用户传进来的字符
    char val;    // 转换后的实际控制值
    struct inode *node = file_inode(file);
    int minor = iminor(node);

    if (size < 1) return -EINVAL;

    /* 1. 从用户空间拷贝 1 个字节的数据 */
    ret = copy_from_user(&status, buf, 1);
    if (ret) return -EFAULT;

    /* 2. 解析数据：字符转数字 (关键修复！！！) */
    if (status == '1') {
        val = 1;
    } else {
        val = 0; // 只要不是 '1'，统统认为是 0 (包括 '0', 换行符等)
    }

    /* 3. 调用下层硬件操作 */
    if (p_led_opr && p_led_opr->ctl) {
        p_led_opr->ctl(minor, val);
    }
    
    /* 4. 告诉内核：这批数据(包括换行符)我都处理完了 (关键修复！！！) */
    return size; 
}

static struct file_operations my_fops = {
    .owner = THIS_MODULE,
    .open  = led_drv_open,
    .write = led_drv_write,
};

/* 核心函数：当下层驱动探测到硬件后，调用此函数注册 */
void register_led_opr(struct led_operations *opr)
{
    int i;
    p_led_opr = opr;
    
    /* 根据下层扫描到的设备数量，创建 /dev/节点 */
    /* 例如：/dev/demo_dev0 (LED), /dev/demo_dev1 (Beep) */
    for (i = 0; i < p_led_opr->num; i++) {
        device_create(my_class, NULL, MKDEV(MAJOR(devid), i), NULL, "demo_dev%d", i);
    }
    printk("Generic Layer: Registered %d devices\n", p_led_opr->num);
}
EXPORT_SYMBOL(register_led_opr);

void unregister_led_opr(void)
{
    int i;
    if (p_led_opr) {
        for (i = 0; i < p_led_opr->num; i++) {
            device_destroy(my_class, MKDEV(MAJOR(devid), i));
        }
        p_led_opr = NULL;
    }
}
EXPORT_SYMBOL(unregister_led_opr);

static int __init led_drv_init(void)
{
    /* 1. 申请设备号 */
    if (alloc_chrdev_region(&devid, 0, MAX_DEVS, DEV_NAME) < 0) {
        printk("Failed to alloc char device region\n");
        return -1;
    }

    /* 2. 初始化 cdev */
    cdev_init(&my_cdev, &my_fops);
    cdev_add(&my_cdev, devid, MAX_DEVS);

    /* 3. 创建类 */
    my_class = class_create(THIS_MODULE, "demo_class");
    
    printk("Generic Layer: Init success\n");
    return 0;
}

static void __exit led_drv_exit(void)
{
    unregister_led_opr(); /* 确保销毁设备节点 */
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(devid, MAX_DEVS);
    printk("Generic Layer: Exit\n");
}

module_init(led_drv_init);
module_exit(led_drv_exit);
MODULE_LICENSE("GPL");