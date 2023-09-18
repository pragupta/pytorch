// !!! This is a file automatically generated by hipify!!!
#pragma once

#ifndef C10_USING_CUSTOM_GENERATED_MACROS

// We have not yet modified the AMD HIP build to generate this file so
// we add an extra option to specifically ignore it.
#ifndef C10_HIP_NO_CMAKE_CONFIGURE_FILE
#include <c10/hip/impl/hip_cmake_macros.h>
#endif // C10_HIP_NO_CMAKE_CONFIGURE_FILE

#endif

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_HIP_BUILD_SHARED_LIBS)
#define C10_HIP_EXPORT __declspec(dllexport)
#define C10_HIP_IMPORT __declspec(dllimport)
#else
#define C10_HIP_EXPORT
#define C10_HIP_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_HIP_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_HIP_EXPORT
#endif // defined(__GNUC__)
#define C10_HIP_IMPORT C10_HIP_EXPORT
#endif // _WIN32

// This one is being used by libc10_hip.so
#ifdef C10_HIP_BUILD_MAIN_LIB
#define C10_HIP_API C10_HIP_EXPORT
#else
#define C10_HIP_API C10_HIP_IMPORT
#endif

/**
 * The maximum number of GPUs that we recognizes.
 */
#define C10_COMPILE_TIME_MAX_GPUS 16
