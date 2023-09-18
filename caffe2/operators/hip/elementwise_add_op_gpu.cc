// !!! This is a file automatically generated by hipify!!!
#include "caffe2/operators/elementwise_add_op.h"

#include "caffe2/core/hip/context_gpu.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Add,
    BinaryElementwiseOp<NumericTypes, HIPContext, AddFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    AddGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        HIPContext,
        AddFunctor<HIPContext>>);

} // namespace caffe2
