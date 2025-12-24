#!/bin/bash

# 1. Download stb_image headers if they don't exist
if [ ! -f src/stb_image.h ]; then
    echo "Downloading stb_image.h..."
    wget -q -O src/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
fi

if [ ! -f src/stb_image_write.h ]; then
    echo "Downloading stb_image_write.h..."
    wget -q -O src/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
fi

# 2. Download a test image if not present
if [ ! -f test_image.png ]; then
    echo "Downloading test image..."
    # A standard Lenna or similar test image
    wget -q -O test_image.png https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
fi

# 3. Build
echo "Building project..."
make

# 4. Run
# Usage: ./gpu_fft <image> <radius>
# Radius 30 is usually a good start for a 512x512 image to see blurring
echo "Running GPU FFT Filter..."
./gpu_fft test_image.png 30

echo "Done! Check output_*.png files."