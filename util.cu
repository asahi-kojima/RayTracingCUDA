#include <curand_kernel.h>
#include "common.h"
#include "util.h"
#include "Math/vector.h"

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

Vec3 RandomGeneratorGPU::generateRandomDirectionOnUnitSphere(const Vec3& normal)
{
	ONB onb(normal);

	const f32 phi = RandomGeneratorGPU::uniform_real() * 2 * M_PI;
	const f32 z = RandomGeneratorGPU::uniform_real(-1, 1);

	const f32 cos0 = sqrtf(1 - z * z + 0.0001f);
	const f32 x = cos(phi) * cos0;
	const f32 y = sin(phi) * cos0;

	return onb.local(x, y, z);
}

Vec3 RandomGeneratorGPU::generateRandomDirectionOnUnitHemiSphere(const Vec3& normal)
{
	ONB onb(normal);

	const f32 phi = RandomGeneratorGPU::uniform_real() * 2 * M_PI;
	const f32 z = RandomGeneratorGPU::uniform_real();

	const f32 cos0 = sqrtf(1 - z * z + 0.0001f);
	const f32 x = cos(phi) * cos0;
	const f32 y = sin(phi) * cos0;

	return onb.local(x, y, z);
}