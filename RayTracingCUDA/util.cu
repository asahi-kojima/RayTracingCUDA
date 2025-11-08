#include <curand_kernel.h>
#include "common.h"
#include "util.h"

extern __device__ curandState s[32];

std::random_device RandomGenerator::rd;
std::mt19937 RandomGenerator::mRandomGenerator;
std::normal_distribution<f32> RandomGenerator::mNormalDist;
std::uniform_real_distribution<f32> RandomGenerator::mUniform0_1Dist;
//
//f32 RandomGeneratorGPU::uniform_real(f32 a, f32 b)
//{
//	const f32 rnd = curand_uniform(&s[(threadIdx.x + blockIdx.x) % 32]);
//	return a + (b - a) * rnd;
//}
//
//f32 RandomGeneratorGPU::signed_uniform_real(f32 a, f32 b)
//{
//	const f32 rnd = curand_uniform(&s[(threadIdx.x + blockIdx.x) % 32]);
//	return a + (b - a) * rnd;
//}
