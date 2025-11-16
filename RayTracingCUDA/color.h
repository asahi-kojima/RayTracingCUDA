#pragma once
#include "common.h"
#include "vector.h"

class Color
{
public:
	__device__ __host__ Color() = default;
	__device__ __host__ Color(const f32 r, const f32 g, const f32 b, const f32 alpha = 1.0f);
	__device__ __host__ Color(const Vec3& rbg, const f32 alpha = 1.0f);
	__device__ __host__ Color(const u32 hexadecimal);

	__device__ __host__ f32 operator[](size_t i) const;
	__device__ __host__ f32& operator[](size_t i);

	__device__ __host__ f32& r();
	__device__ __host__ f32& g();
	__device__ __host__ f32& b();
	__device__ __host__ f32& a();

	__device__ __host__ f32 r() const;
	__device__ __host__ f32 g() const;
	__device__ __host__ f32 b() const;
	__device__ __host__ f32 a() const;

	__device__ __host__ Vec3 getRGB() const { return Vec3(r(), g(), b()); }

	__device__ __host__ Color& operator+=(const Color& rhs);
	__device__ __host__ Color operator*(const Color& rhs) const;
	__device__ __host__ Color operator*(const f32 value) const;
	__device__ __host__ Color& operator*=(const Color& rhs);
	__device__ __host__ Color& operator*=(const f32 value);
	__device__ __host__ Color& operator/=(const f32 value);

	__host__ static Color random();

	__device__ __host__ void clamp();
	__device__ __host__ bool isNan();

	static const Color White;
	static const Color Black;
	static const Color Gray;
	static const Color Red;
	static const Color Green;
	static const Color Lime;
	static const Color Blue;
	static const Color Yellow;
	static const Color Magenta;
	static const Color Orange;
	static const Color Gold;
	static const Color Bronze;
	static const Color Azure;
	static const Color Silver;

private:
	f32 mRGBA[4];
};