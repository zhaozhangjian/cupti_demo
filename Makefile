SOURCE_CODE = cupti_demo.cu
TARGET_BINA = cupti_demo

CC = nvcc

$(TARGET_BINA):$(SOURCE_CODE)
	$(CC) $(SOURCE_CODE) -o $(TARGET_BINA) -lcupti

.PHONY:clean
clean:
	rm -rf $(TARGET_BINA)
