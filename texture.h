#pragma once
#include <memory>
#include "vector.h"
#include "color.h"

class Texture
{
public:
	__device__ __host__ virtual Color color(f32 u, f32 v, const vec3& p) = 0;
};



class ConstantTexture : public Texture
{
public:
	__device__ __host__ ConstantTexture() = default;
	__device__ __host__ ConstantTexture(Color c) : mColor(c) {}

private:
	__device__ __host__ virtual Color color(f32 u, f32 v, const vec3& p) override {return mColor;}

	Color mColor;
};


class CheckerTexture : public Texture
{
public:
	__device__ __host__ CheckerTexture() = default;
	__device__ __host__ CheckerTexture(Color odd_color, Color even_color, const f32 checker_scale = 1.0f)
		: mOddColor(odd_color), mEvecColor(even_color), mCheckerScale(checker_scale)
		{}

private:
	__device__ __host__ virtual Color color(f32 u, f32 v, const vec3& p) override
	{
		const f32 coeff = 2 * M_PI / mCheckerScale;
		const f32 sin_product = sin(p[0] * coeff)* sin(p[1] * coeff)* sin(p[2] * coeff);
		if (sin_product > 0)
		{
			return mEvecColor;
		}
		else
		{
			return mOddColor;
		}
	}

	Color mOddColor;
	Color mEvecColor;
	f32 mCheckerScale;
};


template <class TextureKind, typename... Args>
inline __global__ void make_texture(Texture* p, Args...args)
{
	new (p) TextureKind(args...);
}

template <class TextureKind, typename... Args>
inline Texture* make_texture(Args... args)
{
	Texture* pTexture;
	cudaMalloc(&pTexture, sizeof(TextureKind));
	make_texture<TextureKind><<<1,1>>>(pTexture, args...);

	return pTexture;
}

