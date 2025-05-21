# cupti_demo
This project shows how to use CUPTI to profile CUDA Driver APIs called by a Runtime API.

# Setup
First of all, you need a platform with CUDA installed.

As the cupti_demo.cu from the present project calls some APIs like cudaBindTexture deprecated from CUDA 12, so I use CUDA 11.4.

You can feel free to make modification to the non-CUPTI part as the CUPTI part is what I really want to show you in this project.

# Compile and Run
```
make
sudo ./cupti_demo
```
*sudo* is essential here or you will get an error like:
```
cupti_demo.cu:38: error: function cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES.
```
# Result
You will find messages printed on your screen as shown in example.log, which shows CUDA Runtime and Driver API callings of this application.