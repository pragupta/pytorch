//
//
//   Implementations
//   TODO: bfloat16
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <ATen/native/cuda/f8_hip_impl.cuh>
#include <ATen/Context.h>
#include <limits>
#include <iostream>

#define HIP_HOST_DEVICE __host__ __device__

namespace hip_f8_impl {

__host__ inline int clz(uint32_t x) { return __builtin_clz(x); }
__device__ inline int clz(uint32_t x) { return __clz(x); }

//It seems that we don't need this special treatment anymore. But still keep it
//in case I am wrong.
#if 0
template <int wm, int we, typename T>
HIP_HOST_DEVICE 
uint8_t cast_to_f8_no_range_reduce(T _x, bool stoch = false, uint32_t rng = 0) {
  static_assert(we==5, "we==5");
  static_assert(sizeof(T)==2, "no_range_reduce only works for float16");

  uint32_t x = reinterpret_cast<uint16_t&>(_x);

  uint32_t y, head, mantissa, exponent;
  uint32_t sign;

  const int mfmt = 10;
  head = x & 0xFC00;
  mantissa = x & 0x3FF;
  exponent = (head>>10) & 0x1F;
  sign = head >> 15;
  uint32_t signed_inf = (sign<<7) + (((1<<we)-1)<<wm);

  if((x & 0x7FFF)==0x7C00)
    return signed_inf;
  if((x & 0x7C00)==0x7C00)
    return signed_inf+1;
  if(x==0)
    return 0;
  if(x==0x8000)
    return 0x80;

//  uint32_t nextbit = 1<<(mfmt-wm-1);
  uint32_t drop_mask =  (1 << (mfmt-wm)) - 1;

  int new_exponent = 0, new_mantissa = 0;
  //const int max_exp = (1<<we)-(negative_zero_nan ? 1 : 2);
  mantissa += (stoch ? rng : mantissa) & drop_mask;
  if(exponent!=0)
    mantissa += 1<<mfmt;
  if(mantissa >= (2<<mfmt)) {
    mantissa >>= 1;
    exponent++;
  }
  else if(mantissa>=(1<<mfmt) && exponent==0) {
    exponent++;
  }
  mantissa >>= (mfmt-wm);
  mantissa &= (1<<wm) - 1;
  
  if(exponent == 0 && mantissa == 0)
    return 0;
  if(exponent == 31)
    return (sign << 7) | 0x7B;
  return (sign << 7) | (exponent << wm) | mantissa;
}
#endif

template <int wm, int we, typename T, bool negative_zero_nan, bool clip, bool PRINT_KERNEL_INFO>
HIP_HOST_DEVICE
uint8_t cast_to_f8(T _x, bool stoch, uint32_t rng) {
  constexpr bool is_half = std::is_same<T,__half>::value;
  constexpr bool is_float = std::is_same<T,float>::value;
  static_assert(wm+we==7, "wm+we==7");
  static_assert(is_half || is_float, "Only half and float can be cast to f8");

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id == 0) {
    if (PRINT_KERNEL_INFO == true) {
      printf("f8_hip_impl.cu: cast_to_f8\n");
      printf("\tis_half = %d \n", is_half);
      printf("\tis_float = %d \n", is_float);
    }
  } 

  //if(sizeof(T)==2 && we==5 && !negative_zero_nan)
    //return cast_to_f8_no_range_reduce<2, 5, __half>(_x, stoch, rng);

  const int mfmt = (sizeof(T)==4) ? 23 : 10;
  uint32_t x;
  if(sizeof(T)==4)
    x = reinterpret_cast<uint32_t&>(_x);
  else
    x = reinterpret_cast<uint16_t&>(_x);

  uint32_t y, head, mantissa;
  int exponent, bias;
  uint32_t sign;

  if(sizeof(T)==4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head>>23) & 0xFF;
    sign = head >> 31;
    bias = 127;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head>>10) & 0x1F;
    sign = head >> 15;
    bias = 15;
  }

  uint32_t signed_inf = (sign<<7) + (((1<<we)-1)<<wm);

  // Deal with inf and NaNs
  if(negative_zero_nan) {
    if(sizeof(T)==4) {
      if((x & 0x7F800000) == 0x7F800000)
       return 0x80;
    } else {
      //if(__hisinf(x) || __hisnan(x))
      if((x & 0x7C00)==0x7C00)
       return 0x80;
    }
  }
  else {
    if(sizeof(T)==4) {
      if((x & 0x7F800000) == 0x7F800000)
        return signed_inf + (mantissa!=0 ? 1 : 0);
    } else {
      if((x & 0x7C00)==0x7C00)
        return signed_inf + (mantissa!=0 ? 1 : 0);
    }
  }
  if(x==0)
    return 0;

  // First need to check if it is normal or denorm as there is a difference of implict 1
  // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
  // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for 
  // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
  // exponent and mantissa again

  // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
  const int f8_bias = ( 1<<(we-1) ) - 1  +  ( negative_zero_nan ? 1 : 0 );   
  const int f8_denormal_act_exponent = 1 - f8_bias;  //actual exponent of f8 denormal
  // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
  // f8_exponent is the converted f8 exponent with bias encoding
  // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent, 
  // the difference needs to be adjusted and mantissa shifted
  int act_exponent, f8_exponent, exponent_diff; 

  if (exponent == 0) { // fp32/fp16 is in denormal. 
  /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16 here. 
   In this case, f8 is usually in denormal. But there could be exceptions. 
   fp16 denormal has exponent bias 15 while bf8 with NANOO has exponent bias 16. 
   It means that there are some numbers in fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. 
   fp16 numbers where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal. 
   In this case, the fp16 mantissa should be shift left by 1  */
    act_exponent = exponent - bias + 1;
    exponent_diff = f8_denormal_act_exponent - act_exponent; // actual exponent is exponent-bias+1 as it is denormal
  }
  else { // fp32/fp16 is normal with implicit 1
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) { 
    /* This is the case where fp32/fp16 is normal but it is in f8 denormal range. 
       For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16 
       actual exponent is -7, it is actually larger due to the implict 1, 
       Therefore it needs to be adjust to -6 and mantissa shift right by 1. 
       So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    }
    else { //both fp32/fp16 and f8 are in normal range
      exponent_diff = 0; // exponent_diff=0 does not mean there is no difference for this case, 
                         //act_exponent could be larger. Just that it does not need shift mantissa
    }
    mantissa += (1 << mfmt); //Add the implicit 1 into mantissa
  }


  bool midpoint = (mantissa & ( (1 << (mfmt-wm+exponent_diff)) - 1 )) == ( 1 << (mfmt-wm+exponent_diff-1) ); 
  /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift right 
     as shift right could rip off some residual part and make something not midpoint look like midpoint. 
     For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint, 
     but after shift right by 4 bits, it would look like midpoint.
  */

  if (exponent_diff>0)
    mantissa >>= exponent_diff;
  else if (exponent_diff == -1)
    mantissa <<= -exponent_diff;
  bool implicit_one = mantissa & (1 << mfmt); 
  //if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
  f8_exponent = (act_exponent+exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one?0:1); 

  //Now we have the exponent and mantissa adjusted
  uint32_t drop_mask =  (1 << (mfmt-wm)) - 1;
  //bool midpoint = (mantissa & drop_mask) == ( 1 << (mfmt-wm-1) ); 
  bool odd = mantissa & (1<< (mfmt-wm)); // if the least significant bit that is not truncated is 1
  mantissa += (stoch ? rng : (midpoint?(odd?mantissa:mantissa-1 ) :mantissa) ) & drop_mask;

  //Now we deal with overflow
  if (f8_exponent == 0) {
    if ((1 << mfmt) & mantissa) {
      f8_exponent = 1; //denormal overflow to become normal, promote exponent 
      //mantissa &=  (1<<mfmt) -1 ; //No need to make 1 implicit now as it will be addressed later
    }
  }
  else {
    if ((1 << (mfmt+1)) & mantissa) {
      mantissa >>= 1;
      f8_exponent++;
      //mantissa &=  (1<<mfmt) -1 ; // No need to make 1 implicit now as it will be addressed later
    }
  }
  
  mantissa >>= (mfmt-wm);

  // above range: quantize to maximum possible float of the same sign
  const int max_exp = (1<<we)-(negative_zero_nan ? 1 : 2);
  if(f8_exponent > max_exp) {
    if(clip) {
      mantissa = (1<<wm)-1;
      f8_exponent = max_exp;
    } else {
      return signed_inf;
    }
  }

  if(f8_exponent == 0 && mantissa == 0)
      return negative_zero_nan? 0 : (sign<<7);
  mantissa &= (1<<wm)-1;
  return (sign << 7) | (f8_exponent << wm) | mantissa;
   
}

template <int wm, int we, typename T, bool negative_zero_nan, bool PRINT_KERNEL_INFO>
HIP_HOST_DEVICE
T cast_from_f8(uint8_t x) {
  constexpr bool is_half = std::is_same<T,__half>::value;
  constexpr bool is_float = std::is_same<T,float>::value;
  constexpr bool is_bf16 = std::is_same<T,hip_bfloat16>::value;
  static_assert(is_half || is_float, "only half and float are supported");

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id == 0) {
    if (PRINT_KERNEL_INFO == true) {
      printf("f8_hip_impl.cu: cast_from_f8\n");
      printf("\tis_half = %d \n", is_half);
      printf("\tis_float = %d \n", is_float);
    }
  }

  constexpr int weo = is_half ? 5 : 8;
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

  T fInf, fNegInf, fNaN, fNeg0;
  if(is_half) {
    const uint16_t ihInf = 0x7C00;
    const uint16_t ihNegInf = 0xFC00;
    const uint16_t ihNaN = 0x7C01;
    const uint16_t ihNeg0 = 0x8000;
    fInf = reinterpret_cast<const T&>(ihInf);
    fNegInf = reinterpret_cast<const T&>(ihNegInf);
    fNaN = reinterpret_cast<const T&>(ihNaN);
    fNeg0 = reinterpret_cast<const T&>(ihNeg0);
  } else if(is_float) {
    const uint32_t ifInf = 0x7F800000;
    const uint32_t ifNegInf = 0xFF800000;
    const uint32_t ifNaN = 0x7F800001;
    const uint32_t ifNeg0 = 0x80000000;
    fInf = reinterpret_cast<const T&>(ifInf);
    fNegInf = reinterpret_cast<const T&>(ifNegInf);
    fNaN = reinterpret_cast<const T&>(ifNaN);
    fNeg0 = reinterpret_cast<const T&>(ifNeg0);
  }

  if(x==0){
  
    if(is_half){
		const uint16_t retval = 0x0000;
		return reinterpret_cast<const T&>(retval);
	}
	else if(is_float)
	{
		const uint32_t retval = 0x00000000;
		return reinterpret_cast<const T&>(retval);
	}

  }
  //	return 0;
  //  return static_cast<const T&>(x);

  uint32_t sign = x>>7;
  uint32_t mantissa = x & ((1<<wm)-1);
  int exponent = (x & 0x7F) >> wm;
  if(negative_zero_nan) {
    if(x==0x80)
      return fNaN;
  } else {
    if(x==0x80) 
      return fNeg0;
    if(exponent == ((1<<we)-1))
       return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
  }
  typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type retval;
  if(we==5 && is_half && !negative_zero_nan) {
     retval = x<<8;
     return reinterpret_cast<const T&>(retval);
  }

  const int exp_low_cutoff = (1<<(weo-1)) - (1<<(we-1)) + 1 - (negative_zero_nan ? 1 : 0);

  //subnormal input
  if(exponent == 0) {
    //guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + clz(mantissa) - (32-wm);
    mantissa <<= sh;
    exponent += 1-sh;
    /*
    exponent++;
    while(mantissa<(1<<wm)) {
      mantissa <<= 1;
      exponent--;
    }
    */
    mantissa &= ((1<<wm)-1);
  }
  exponent += exp_low_cutoff-1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if(exponent<=0) {
    mantissa |= 1<<wmo;
    mantissa >>= 1-exponent;
    exponent = 0;
  }

  if(sizeof(T)==2)
    retval = (sign<<15) | (exponent<<10) | mantissa;
  else
    retval = (sign<<31) | (exponent<<23) | mantissa;
  return reinterpret_cast<const T&>(retval);
}


} // namespace hip_f8_impl

template <typename T, int we, int wm, bool PRINT_KERNEL_INFO>
__global__ void Quant8_inplace(T* _p, int32_t count, bool stoch, uint32_t seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i == 0) {
    if (PRINT_KERNEL_INFO == true) {
      printf("f8_hip_impl.cu: Quant8_inplace<, %d, %d, %d>\n", we, wm, PRINT_KERNEL_INFO);
    }
  }
  if (i >= count) return;
  typedef typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type IT;
  typedef typename std::conditional<sizeof(T)==2, __half, float>::type FT;
  IT* p = (IT*) _p;
  FT* fp = (FT*) _p;
  IT x = p[i];
//  const int we=5, wm=2;

  uint8_t y;
  if(!stoch)
    y = hip_f8_impl::cast_to_f8<wm,we,FT,false,true,PRINT_KERNEL_INFO>(fp[i], false, 0);
  else {
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    if(sizeof(x)==4)
      drop_bits ^= x>>16;
    drop_bits = ((drop_bits & 31)<<11) | (drop_bits>>5);
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (i*229791) ^ seed);
    y = hip_f8_impl::cast_to_f8<wm,we,FT,false,true,PRINT_KERNEL_INFO>(fp[i], true, rng);
  }
  fp[i] = hip_f8_impl::cast_from_f8<wm,we,FT,false,PRINT_KERNEL_INFO>(y);
}


template __global__ void Quant8_inplace<__half,5,2,true>(__half* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<__half,5,2,false>(__half* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<__half,4,3,true>(__half* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<__half,4,3,false>(__half* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<float,5,2,true>(float* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<float,5,2,false>(float* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<float,4,3,true>(float* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_inplace<float,4,3,false>(float* _p, int32_t count, bool stoch, uint32_t seed);


void Quant8_inplace_host(__half* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152) {
        auto fun = f152 ? Quant8_inplace<__half,5,2,false> : Quant8_inplace<__half,4,3,false>;
        if (at::globalContext().allowF8ROCMLOG()) {
                std::cout << "f8_rocm_helper: Quant8_inplace_host(__half* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152)" << std::endl;
                fun = f152 ? Quant8_inplace<__half,5,2,true> : Quant8_inplace<__half,4,3,true>;
        }
        uint32_t dim_a = count; 
        uint32_t grid_a = (dim_a+255)/256;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(fun),
           dim3(grid_a,1,1), dim3(256,1,1), 0, stream, _p, dim_a, true, seed);
}

void Quant8_inplace_host(float* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152) {
        auto fun = f152 ? Quant8_inplace<float,5,2,false> : Quant8_inplace<float,4,3,false>;
        if (at::globalContext().allowF8ROCMLOG()) {
                std::cout << "f8_rocm_helper: Quant8_inplace_host(float* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152) " << std::endl;
                fun = f152 ? Quant8_inplace<float,5,2,true> : Quant8_inplace<float,4,3,true>;
        } 
        uint32_t dim_a = count; 
        uint32_t grid_a = (dim_a+255)/256;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(fun),
           dim3(grid_a,1,1), dim3(256,1,1), 0, stream, _p, dim_a, true, seed);
}
