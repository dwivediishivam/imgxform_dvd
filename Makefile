# dvd.shivam: Makefile for imgxform_dvd
NVCC := nvcc
TARGET := imgxform_dvd.out
SRCS := imgxform_dvd_main.cu imgxform_dvd_kernels.cu imgxform_dvd_utils.cpp imgxform_dvd_ppmio.cpp

all: $(TARGET)  # dvd.shivam

$(TARGET): $(SRCS)
	$(NVCC) -std=c++11 -O2 -o $(TARGET) $(SRCS) -lnppif -lnppc -lnppisu

clean:
	rm -f $(TARGET)