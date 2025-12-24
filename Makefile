NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -I./src
LDFLAGS = -lcufft

SRCS = src/image_fft.cu src/kernels.cu
OBJS = $(SRCS:.cu=.o)
TARGET = gpu_fft

HEADERS = src/stb_image.h src/stb_image_write.h

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

src/stb_image.h:
	wget -q -O $@ https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

src/stb_image_write.h:
	wget -q -O $@ https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

clean:
	rm -f $(OBJS) $(TARGET) output_*.png