// !!! This is a file automatically generated by hipify!!!
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/sgd/learning_rate_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(LearningRate, LearningRateOp<float, HIPContext>);
}  // namespace caffe2
