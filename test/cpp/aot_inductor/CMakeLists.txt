message(WARNING "PG in aot_inductor cmake with USE_ROCM=${USE_ROCM}")
set(AOT_INDUCTOR_TEST_ROOT ${TORCH_ROOT}/test/cpp/aot_inductor)

# Build the cpp gtest binary containing the cpp-only tests.
if (USE_CUDA)
  set(INDUCTOR_TEST_SRCS
    ${AOT_INDUCTOR_TEST_ROOT}/test.cpp
  )
elseif (USE_ROCM)
  set(INDUCTOR_TEST_SRCS
    ${AOT_INDUCTOR_TEST_ROOT}/hip/test.cpp
  )
endif()


add_executable(test_aot_inductor
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${INDUCTOR_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aot_inductor PRIVATE USE_GTEST)

# Define a custom command to generate the library
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libaot_inductor_output.so
        COMMAND python ${AOT_INDUCTOR_TEST_ROOT}/test.py
        DEPENDS ${AOT_INDUCTOR_TEST_ROOT}/test.py
)
add_custom_target(aot_inductor_output_target ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libaot_inductor_output.so)
add_dependencies(test_aot_inductor aot_inductor_output_target)

target_link_libraries(test_aot_inductor PRIVATE
  torch
  gtest
  ${CMAKE_CURRENT_BINARY_DIR}/libaot_inductor_output.so
)

target_include_directories(test_aot_inductor PRIVATE ${ATen_CUDA_INCLUDE})

if(USE_CUDA)
  target_link_libraries(test_aot_inductor PRIVATE
    ${C10_CUDA_BUILD_SHARED_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES}
  )


  target_compile_definitions(test_aot_inductor PRIVATE USE_CUDA)
elseif(USE_ROCM)
    message(WARNING "PG in aot_inductor inside USE_ROCM=${USE_ROCM} branch")
    target_link_libraries(test_aot_inductor PRIVATE
        ${C10_HIP_BUILD_SHARED_LIBS}
        ${ROCM_HIPRTC_LIB}
        ${PYTORCH_HIP_HCC_LIBRARIES}
        ${TORCH_CUDA_LIBRARIES}
        )
    target_compile_definitions(test_aot_inductor PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  install(TARGETS test_aot_inductor DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aot_inductor> DESTINATION bin OPTIONAL)
  endif()
endif()
