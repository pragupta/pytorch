#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#include <roctracer/roctx.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
  nvtx.def("rangePushA", roctxRangePushA);
  nvtx.def("rangePop", roctxRangePop);
  nvtx.def("rangeStartA", roctxRangeStartA);
  nvtx.def("rangeEnd", roctxRangeStop);
  nvtx.def("markA", roctxMarkA);
}

} // namespace torch::cuda::shared
