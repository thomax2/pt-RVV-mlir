#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/dma-buf.h>
#include <linux/file.h>
#include <linux/mm.h>
// #include <linux/cache.h>
#include <asm/io.h> // 提供 virt_to_phys
// #include <asm/pgtable.h>

#define DMABUF_EXP_NAME "dmabuf_exporter"

// 适配 FEC 接收环：2048 字节 * 64 个描述符 = 128KB
#define DMABUF_SIZE (2048 * 512) 

// 自定义 IOCTL 命令
#define DMABUF_IOCTL_ALLOC   _IOWR('D', 1, int)
#define DMABUF_IOCTL_GET_PHY _IOWR('D', 2, unsigned int) // 新增：获取物理地址

// 全局变量：用于临时保存上一次分配的物理地址供测试程序获取
static unsigned int last_allocated_paddr = 0;

// 我们的私有结构体，附在 dma_buf->priv 上
struct my_dmabuf_priv {
    void *vaddr;         // 内核虚拟地址
    dma_addr_t paddr;    // 物理地址
    int order;           // 物理页阶数
};

// --- dma_buf 操作函数集 ---

// 1. 映射到 CPU (mmap)
static int my_dmabuf_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
    struct my_dmabuf_priv *priv = dmabuf->priv;

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    
    // 将我们申请的连续物理内存映射给用户空间
    // 使用 paddr >> PAGE_SHIFT 获取页帧号 (PFN)
    return remap_pfn_range(vma, vma->vm_start, 
                           priv->paddr >> PAGE_SHIFT,
                           vma->vm_end - vma->vm_start, 
                           vma->vm_page_prot);
}

// 2. 释放 Buffer
static void my_dmabuf_release(struct dma_buf *dmabuf)
{
    struct my_dmabuf_priv *priv = dmabuf->priv;
    if (priv) {
        // 使用 free_pages 释放多页连续内存
        free_pages((unsigned long)priv->vaddr, priv->order); 
        kfree(priv);
    }
    printk(KERN_INFO "DMABUF: Buffer released\n");
}

// 3. 其他必须实现的桩函数 (本例简单演示，暂不涉及多设备共享，故留空)
static struct sg_table *my_dmabuf_map(struct dma_buf_attachment *attachment, enum dma_data_direction dir) { return NULL; }
static void my_dmabuf_unmap(struct dma_buf_attachment *attachment, struct sg_table *table, enum dma_data_direction dir) {}
static int my_dmabuf_attach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment) { return 0; }
static void my_dmabuf_detach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment) {}

static const struct dma_buf_ops my_dmabuf_ops = {
    .attach = my_dmabuf_attach,
    .detach = my_dmabuf_detach,
    .map_dma_buf = my_dmabuf_map,
    .unmap_dma_buf = my_dmabuf_unmap,
    .release = my_dmabuf_release,
    .mmap = my_dmabuf_mmap, // 关键：允许用户 mmap
};

// --- Misc 设备操作 ---

static long my_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct my_dmabuf_priv *priv;
    struct dma_buf_export_info exp_info = {0};
    struct dma_buf *dmabuf;
    int fd;

    // 新增分支：处理用户态获取物理地址的请求
    if (cmd == DMABUF_IOCTL_GET_PHY) {
        if (copy_to_user((unsigned int __user *)arg, &last_allocated_paddr, sizeof(last_allocated_paddr)))
            return -EFAULT;
        return 0;
    }

    if (cmd != DMABUF_IOCTL_ALLOC) return -EINVAL;

    priv = kzalloc(sizeof(*priv), GFP_KERNEL);
    if (!priv) return -ENOMEM;

    // 核心修改：使用 __get_free_pages 分配连续的 128KB 物理内存
    // get_order 自动计算需要 2 的多少次幂个物理页 (128KB 对应 order = 5, 即 32 个 4KB页)
    priv->order = get_order(DMABUF_SIZE);
    priv->vaddr = (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, priv->order);
    if (!priv->vaddr) {
        kfree(priv);
        return -ENOMEM;
    }
    
    // 获取内核虚拟地址对应的真实物理地址
    priv->paddr = virt_to_phys(priv->vaddr);
    last_allocated_paddr = (unsigned int)priv->paddr;

    exp_info.ops = &my_dmabuf_ops;
    exp_info.size = DMABUF_SIZE;
    exp_info.flags = O_CLOEXEC | O_RDWR;
    exp_info.priv = priv;

    dmabuf = dma_buf_export(&exp_info);
    if (IS_ERR(dmabuf)) {
        free_pages((unsigned long)priv->vaddr, priv->order);
        kfree(priv);
        return PTR_ERR(dmabuf);
    }

    fd = dma_buf_fd(dmabuf, O_CLOEXEC);
    if (fd < 0) {
        dma_buf_put(dmabuf);
        return fd;
    }

    printk(KERN_INFO "DMABUF: Allocated %d Bytes, Order %d, PAddr: 0x%08x, fd: %d\n", 
           DMABUF_SIZE, priv->order, last_allocated_paddr, fd);

    if (copy_to_user((int __user *)arg, &fd, sizeof(fd))) {
        return -EFAULT;
    }

    return 0;
}

static const struct file_operations my_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = my_ioctl,
};

static struct miscdevice my_misc_dev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = DMABUF_EXP_NAME,
    .fops = &my_fops,
};

static int __init my_init(void)
{
    return misc_register(&my_misc_dev);
}

static void __exit my_exit(void)
{
    misc_deregister(&my_misc_dev);
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");