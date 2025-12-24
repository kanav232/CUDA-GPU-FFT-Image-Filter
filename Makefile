NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -I./src
LDFLAGS = -lcufft

SRCS = src/image_fft.cu src/kernels.cu
OBJS = $(SRCS:.cu=.o)
TARGET = gpu_fft

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) output_*.png