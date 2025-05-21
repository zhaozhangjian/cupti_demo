#include <stdio.h>
#include "cuda_texture_types.h"
#include <cuda_runtime.h>
#include <cupti.h>

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
        __FILE__, __LINE__, #call, errstr);                                \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

static CUpti_SubscriberHandle cuptiSubscriber;

void cuptiSubscriberCallback(
  void *userdata,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cb_id,
  const CUpti_CallbackData *cb_info) {
  const char* apiName;
  cuptiGetCallbackName(domain, cb_id, &apiName);
  printf("callback: %d %d %d %p %s\n", domain, cb_id, cb_info->correlationId, cb_info, apiName);
}

void initTrace() {
  // Subscribe callbacks
  CUPTI_CALL(cuptiSubscribe(&cuptiSubscriber, (CUpti_CallbackFunc) cuptiSubscriberCallback, (void *) NULL));
  CUPTI_CALL(cuptiEnableDomain(1, cuptiSubscriber, CUPTI_CB_DOMAIN_DRIVER_API));
  CUPTI_CALL(cuptiEnableDomain(1, cuptiSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
}

void finiTrace() {
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  CUPTI_CALL(cuptiUnsubscribe(cuptiSubscriber));
}

// Texture reference for 1D float texture
texture<float, 1, cudaReadModeElementType> texRefUniqueName;

// Kernel that uses texture memory
__global__ void textureKernel(float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < width) {
      // Read from texture memory
      output[x] = tex1Dfetch(texRefUniqueName, x);
    }
}
// Kernel that NOT uses texture memory
__global__ void noTextureKernel(float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < width) {
      // Read from memory
      output[x] = output[x] + (float)x;
    }
}

int main() {
  const int N = 64;
  const textureReference* texref = nullptr;
  float* h_data = (float*)malloc(N * sizeof(float));
  float* h_output = (float*)malloc(N * sizeof(float));
  
  // Initialize host data
  for (int i = 0; i < N; i++) {
    h_data[i] = (float)i;
  }

  initTrace();

  // Allocate device memory
  float* d_data;
  float* d_output;
  cudaMalloc((void**)&d_data, N * sizeof(float));
  cudaMalloc((void**)&d_output, N * sizeof(float));
  
  // Copy data to device
  cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

  // Bind texture to device memory
  cudaBindTexture(NULL, texRefUniqueName, d_data, N * sizeof(float));
  cudaGetTextureReference((const textureReference**)&texref, (void*)&texRefUniqueName);
  
  // Launch kernel
  dim3 blockDim(16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  textureKernel<<<gridDim, blockDim>>>(d_output, N);
  
  noTextureKernel<<<gridDim, blockDim>>>(d_output, N);
  
  // Copy results back to host
  cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Verify results
  bool success = true;
  for (int i = 0; i < N; i++) {
    float golden = h_data[i] * 2.0f;
    if (h_output[i] != golden) {
      printf("Error at %d: %f != %f\n", i, h_output[i], golden);
      success = false;
    }
  }

  if (success) {
    printf("Texture memory test passed!\n");
  }

  // Cleanup
  cudaUnbindTexture(texRefUniqueName);
  cudaFree(d_data);
  cudaFree(d_output);
  free(h_data);
  free(h_output);

  finiTrace();

  return 0;
}
