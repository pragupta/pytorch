
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

AOTITorchError aoti_torch_create_cuda_stream_guard(
    CUDAStreamGuardHandle* ret_guard,
    void* stream,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::hip::HIPStreamGuardMasqueradingAsCUDA* guard =
        new at::hip::HIPStreamGuardMasqueradingAsCUDA(at::hip::getStreamFromExternalMasqueradingAsCUDA(
            static_cast<hipStream_t>(stream), device_index));
    *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::hip::HIPStreamGuardMasqueradingAsCUDA*>(guard); });
}
