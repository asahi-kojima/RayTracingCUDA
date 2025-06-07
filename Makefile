NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAG=-std=c++17 -rdc=true -O3 -DNDEBUG -w -I . -gencode arch=compute_120,code=sm_120


SRC_DIRS = . Math Object
BUILD_DIR = ./build
VPATH = $(SRC_DIRS)
TARGET=RayTracerExe.o

SRCS=$(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cu))
OBJS=$(patsubst %.cu, $(BUILD_DIR)/%.o, $(notdir $(SRCS)))

#ビルド用ディレクトリ作成のためのダミー変数
DUMMY_MKDIR:=$(shell mkdir -p build)



all: $(TARGET)
	./$(TARGET) 1920 1080 > log.txt
	convert ./picture/result.ppm ./picture/result.png

compile: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAG) -o $(TARGET) $(OBJS)

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAG) -c -o $@ $<


clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all compile clean