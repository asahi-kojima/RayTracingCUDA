#pragma once
#include "common.h"

//https://learn.microsoft.com/ja-jp/windows/win32/numerics_h/float3-structure

class Vec3
{
public:
	__device__ __host__ Vec3() = default;
	__device__ __host__ Vec3(f32 x, f32 y, f32 z) : mElements{ x, y, z } {}
	__device__ __host__ Vec3(const Vec3& v) : mElements{ v[0], v[1], v[2] } {}
	__device__ __host__ Vec3(const float3& v) : mElements{ v.x, v.y, v.z } {}

	__device__ __host__  f32 x() const { return mElements[0]; }
	__device__ __host__  f32 y() const { return mElements[1]; }
	__device__ __host__  f32 z() const { return mElements[2]; }
	__device__ __host__  f32& x() { return mElements[0]; }
	__device__ __host__  f32& y() { return mElements[1]; }
	__device__ __host__  f32& z() { return mElements[2]; }

	__device__ __host__  f32& operator[](size_t i);
	__device__ __host__  f32 operator[](size_t i) const;

	__device__ __host__ Vec3 operator-() const;

	__device__ __host__ Vec3  operator+(const Vec3&) const;
	__device__ __host__ Vec3& operator+=(const Vec3&);
	__device__ __host__ Vec3  operator+(const f32)   const;
	__device__ __host__ Vec3& operator+=(const f32);
	__device__ __host__ Vec3  operator-(const Vec3&) const;
	__device__ __host__ Vec3& operator-=(const Vec3&);
	__device__ __host__ Vec3  operator-(const f32)   const;
	__device__ __host__ Vec3& operator-=(const f32);
	__device__ __host__ Vec3  operator*(const Vec3&) const;
	__device__ __host__ Vec3& operator*=(const Vec3&);
	__device__ __host__ Vec3  operator*(const f32)   const;
	__device__ __host__ Vec3& operator*=(const f32);
	__device__ __host__ Vec3  operator/(const f32)   const;
	__device__ __host__ Vec3& operator/=(const f32);

	__device__ __host__ Vec3 normalize() const;
	__device__ __host__ static Vec3 normalize(const Vec3&);

	__device__ __host__ f32 length() const;
	__device__ __host__ static f32 length(const Vec3&);
	__device__ __host__ f32 lengthSquared() const;
	__device__ __host__ static f32 lengthSquared(const Vec3&);

	__device__ __host__ static f32 dot(const Vec3&, const Vec3&);
	__device__ __host__ static Vec3 cross(const Vec3&, const Vec3&);
	__device__ __host__ static Vec3 reflect(const Vec3& v, const Vec3& n);
	__device__ __host__ static Vec3 generateRandomUnitVector();

	__device__ __host__ static inline Vec3 zero() { return Vec3(0.0f, 0.0f, 0.0f); };
	__device__ __host__ static inline Vec3 one() { return Vec3(1.0f, 1.0f, 1.0f).normalize(); };
	__device__ __host__ static inline Vec3 unitX() { return Vec3(1.0f, 0.0f, 0.0f); };
	__device__ __host__ static inline Vec3 unitY() { return Vec3(0.0f, 1.0f, 0.0f); };
	__device__ __host__ static inline Vec3 unitZ() { return Vec3(0.0f, 0.0f, 1.0f); };

	__device__ __host__ bool isNan() const;

#ifdef _DEBUG
	__device__ __host__ void debugPrint(const char* message = "") const;
#endif

private:
	f32 mElements[3];
};

__device__ __host__ Vec3 operator*(const f32 value, const Vec3& v);


class Vec4
{
public:
	__device__ __host__ Vec4(f32 x = 0, f32 y = 0, f32 z = 0, f32 w = 0) : mElements{ x, y, z, w } {}
	__device__ __host__ Vec4(f32* vec4AsArray) : mElements{ vec4AsArray[0], vec4AsArray[1], vec4AsArray[2], vec4AsArray[3] } {}
	__device__ __host__ Vec4(const Vec4& v) : mElements{ v[0], v[1], v[2], v[3] } {}
	__device__ __host__ Vec4(const Vec3& xyz, f32 w) : mElements{ xyz[0], xyz[1], xyz[2], w } {}

	__device__ __host__  f32 x() const { return mElements[0]; }
	__device__ __host__  f32 y() const { return mElements[1]; }
	__device__ __host__  f32 z() const { return mElements[2]; }
	__device__ __host__  f32 w() const { return mElements[3]; }
	__device__ __host__  f32& x() { return mElements[0]; }
	__device__ __host__  f32& y() { return mElements[1]; }
	__device__ __host__  f32& z() { return mElements[2]; }
	__device__ __host__  f32& w() { return mElements[3]; }


	__device__ __host__  f32& operator[](size_t i);
	__device__ __host__  f32  operator[](size_t i) const;

	//__device__ __host__ Vec4 operator-() const;

	__device__ __host__ Vec4  operator+(const Vec4&) const;
	__device__ __host__ Vec4& operator+=(const Vec4&);
	__device__ __host__ Vec4  operator+(const f32) const;
	__device__ __host__ Vec4& operator+=(const f32);
	__device__ __host__ Vec4  operator-(const Vec4&) const;
	__device__ __host__ Vec4& operator-=(const Vec4&);
	__device__ __host__ Vec4  operator-(const f32) const;
	__device__ __host__ Vec4& operator-=(const f32);
	__device__ __host__ Vec4  operator*(const Vec4&) const;
	__device__ __host__ Vec4& operator*=(const Vec4&);
	__device__ __host__ Vec4  operator*(const f32) const;
	__device__ __host__ Vec4& operator*=(const f32);
	__device__ __host__ Vec4  operator/(const f32) const;
	__device__ __host__ Vec4& operator/=(const f32);

	__device__ __host__ Vec3 extractXYZ() const;

private:
	f32 mElements[4];
};