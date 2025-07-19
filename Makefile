# NVCC=/usr/bin/nvcc
NVCC=/usr/local/cuda/bin/nvcc
OPTIMIZE= -O3 -Xptxas -O3 
NVCCFLAG=-std=c++17 -rdc=true $(OPTIMIZE) -DNDEBUG -w -I . -gencode arch=compute_120,code=sm_120

SRC_DIRS = . Math Object
BUILD_DIR = ./build
VPATH = $(SRC_DIRS)
TARGET=RayTracerExe.o

SRCS=$(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cu))
OBJS=$(patsubst %.cu, $(BUILD_DIR)/%.o, $(notdir $(SRCS)))
DEPS:=$(OBJS:%.o=%.d)

#ビルド用ディレクトリ作成のためのダミー変数
DUMMY_MKDIR:=$(shell mkdir -p build)



all: $(TARGET)
	./$(TARGET) 1920 1080 30 20 > log.txt
	convert ./picture/result.ppm ./picture/result.png

release: $(TARGET)
	./$(TARGET) 3840 2160 200 30 > log.txt
	convert ./picture/result.ppm ./picture/hegh-resolution.png

test:  $(TARGET)
	./$(TARGET) 500 500 200 20 > log.txt
	convert ./picture/result.ppm ./picture/result.png

compile: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAG) -o $(TARGET) $(OBJS)

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAG) -MMD -c -o $@ $<

-include $(DEPS)

clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

.PHONY: all compile clean