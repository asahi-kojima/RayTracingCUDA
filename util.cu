#include <curand_kernel.h>
#include "common.h"
#include "util.h"

extern __device__ curandState s[32];

f32 RandomGeneratorGPU::uniform_real(f32 a, f32 b)
{
	const f32 rnd = curand_uniform(&s[(threadIdx.x + blockIdx.x) % 32]);
	return a + (b - a) * rnd;
}

f32 RandomGeneratorGPU::signed_uniform_real(f32 a, f32 b)
{
	const f32 rnd = curand_uniform(&s[(threadIdx.x + blockIdx.x) % 32]);
	return a + (b - a) * rnd;
}
