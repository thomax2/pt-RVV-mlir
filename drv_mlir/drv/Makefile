KERN_DIR = /home/thomax/embed/linux-imx
BUILD_DIR = $(CURDIR)/build
ARCH = arm
CROSS_COMPILE = arm-linux-gnueabihf-

all:
	@mkdir -p $(BUILD_DIR)
	@cp -f *.c *.h $(BUILD_DIR)/ 2>/dev/null || true
	@echo "obj-m += led_drv.o gpio.o dmabuf_exp.o" > $(BUILD_DIR)/Makefile
	$(MAKE) -C $(KERN_DIR) M=$(BUILD_DIR) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) modules
	@cp $(BUILD_DIR)/*.ko $(CURDIR)/ 2>/dev/null || true

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.ko *.o *.mod.c *.mod.o Module.symvers modules.order

.PHONY: all clean