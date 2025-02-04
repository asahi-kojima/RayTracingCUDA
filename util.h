#pragma once
#include <random>
#include "typeinfo.h"



#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
#define GPU_ERROR_CHECKER(code) {gpuAssert((code), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char* file, u32 line)
{
	if (code != cudaSuccess)
	{
		printf("GPU assert : Code %d : %d line in %s", code, line, file);
	}
}

class RandomGenerator
{
public:
	static f32 uniform_real(f32 a = 0.0f, f32 b = 1.0f)
	{
		return a + (b - a) * mUniform0_1Dist(mRandomGenerator);
	}

	static f32 signed_uniform_real(f32 a = -1.0f, f32 b = 1.0f)
	{
		return a + (b - a) * mUniform0_1Dist(mRandomGenerator);
	}

	static f32 normal(f32 mu = 0.0f, f32 sigma = 1.0f)
	{
		return mu + sigma * mNormalDist(mRandomGenerator);
	}

	static s32 uniform_int(s32 a = 0, s32 b = 1)
	{
		f32 rand = uniform_real(static_cast<f32>(a), static_cast<f32>(b));

		return static_cast<s32>(rand);
	}

	static s32 uniform_int_round(s32 center = 0, s32 radius = 1)
	{
		f32 rand = uniform_real(static_cast<f32>(center - radius), static_cast<f32>(center + radius));

		return static_cast<s32>(rand);
	}

	static std::random_device rd;
	static std::mt19937 mRandomGenerator;
	static std::normal_distribution<f32> mNormalDist;
	static std::uniform_real_distribution<f32> mUniform0_1Dist;
};

inline std::random_device RandomGenerator::rd;
inline std::mt19937 RandomGenerator::mRandomGenerator(rd());
inline std::normal_distribution<float> RandomGenerator::mNormalDist(0.0f, 1.0f);
inline std::uniform_real_distribution<f32> RandomGenerator::mUniform0_1Dist(0.0f, 1.0f);


class RandomGeneratorGPU
{
public:
	__device__ static f32 uniform_real(f32 a = 0.0f, f32 b = 1.0f);
	__device__ static f32 signed_uniform_real(f32 a = -1.0f, f32 b = 1.0f);
};


class Managed
{
public:
	void* operator new(size_t allocate_size)
	{
		void* p;
		cudaMallocManaged((void**)&p, allocate_size);
		cudaDeviceSynchronize();

		return p;
	}

	void operator delete(void* p)
	{
		cudaDeviceSynchronize();
		cudaFree(p);
	}
};