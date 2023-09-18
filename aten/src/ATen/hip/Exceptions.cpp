// !!! This is a file automatically generated by hipify!!!
//NS: HIPCachingAllocator must be included before to get CUDART_VERSION definedi
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>

#include <ATen/hip/Exceptions.h>

namespace at::cuda {
namespace blas {

C10_EXPORT const char* _cublasGetErrorEnum(hipblasStatus_t error) {
  if (error == HIPBLAS_STATUS_SUCCESS) {
    return "HIPBLAS_STATUS_SUCCESS";
  }
  if (error == HIPBLAS_STATUS_NOT_INITIALIZED) {
    return "HIPBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == HIPBLAS_STATUS_ALLOC_FAILED) {
    return "HIPBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == HIPBLAS_STATUS_INVALID_VALUE) {
    return "HIPBLAS_STATUS_INVALID_VALUE";
  }
  if (error == HIPBLAS_STATUS_ARCH_MISMATCH) {
    return "HIPBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == HIPBLAS_STATUS_MAPPING_ERROR) {
    return "HIPBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == HIPBLAS_STATUS_EXECUTION_FAILED) {
    return "HIPBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == HIPBLAS_STATUS_INTERNAL_ERROR) {
    return "HIPBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == HIPBLAS_STATUS_NOT_SUPPORTED) {
    return "HIPBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

} // namespace blas

#ifdef CUDART_VERSION
namespace solver {

C10_EXPORT const char* cusolverGetErrorMessage(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:             return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:                return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:               return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:               return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:            return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:              return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:                                          return "Unknown cusolver error number";
  }
}

} // namespace solver
#endif

} // namespace at::cuda
