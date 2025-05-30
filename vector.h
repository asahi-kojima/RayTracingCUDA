#pragma once
#include <cassert>
#include "common.h"

class vec3
{
public:
	__device__ __host__ vec3(f32 x = 0, f32 y = 0, f32 z = 0) : mElements{ x, y, z } {}
	__device__ __host__ vec3(const vec3& v) : mElements{v[0], v[1], v[2]} {}

	__device__ __host__  f32 x() const { return mElements[0]; }
	__device__ __host__  f32 y() const { return mElements[1]; }
	__device__ __host__  f32 z() const { return mElements[2]; }
	__device__ __host__  f32& x() { return mElements[0]; }
	__device__ __host__  f32& y() { return mElements[1]; }
	__device__ __host__  f32& z() { return mElements[2]; }

	__device__ __host__  f32& operator[](size_t i);
	__device__ __host__  f32 operator[](size_t i) const;

	__device__ __host__ vec3 operator-() const;

	__device__ __host__ vec3 operator+(const vec3&) const;
	__device__ __host__ vec3& operator+=(const vec3&);
	__device__ __host__ vec3 operator+(const f32) const;
	__device__ __host__ vec3& operator+=(const f32);
	__device__ __host__ vec3 operator-(const vec3&) const;
	__device__ __host__ vec3& operator-=(const vec3&);
	__device__ __host__ vec3 operator-(const f32) const;
	__device__ __host__ vec3& operator-=(const f32);
	__device__ __host__ vec3 operator*(const vec3&) const;
	__device__ __host__ vec3& operator*=(const vec3&);
	__device__ __host__ vec3 operator*(const f32) const;
	__device__ __host__ vec3& operator*=(const f32);
	__device__ __host__ vec3 operator/(const f32) const;
	__device__ __host__ vec3& operator/=(const f32);

	__device__ __host__ vec3 normalize() const;
	__device__ __host__ static vec3 normalize(const vec3&);

	__device__ __host__ f32 length() const;
	__device__ __host__ static f32 length(const vec3&);
	__device__ __host__ f32 lengthSquared() const;
	__device__ __host__ static f32 lengthSquared(const vec3&);

	__device__ __host__ static f32 dot(const vec3&, const vec3&);
	__device__ __host__ static vec3 cross(const vec3&, const vec3&);
	__device__ __host__ static vec3 reflect(const vec3& v, const vec3& n);
	__device__ __host__ static vec3 generateRandomUnitVector();
	
	__device__ __host__ static inline vec3 zero() { return vec3(0.0f, 0.0f, 0.0f); };
	__device__ __host__ static inline vec3 one() { return vec3(1.0f, 1.0f, 1.0f); };

	__device__ __host__ void print_debug() const;

private:
	f32 mElements[3];
};


__device__ __host__ vec3 operator*(const f32 value, const vec3& v);
__device__ __host__ vec3 normalize(const vec3& v);
__device__ __host__ vec3 cross(const vec3& v0, const vec3& v1);
__device__ __host__ f32 dot(const vec3& v0, const vec3& v1);
__device__ __host__ vec3 reflect(const vec3& v, const vec3& n);

__device__ vec3 random_in_unit_sphere();


class Vec4
{
public:
	__device__ __host__ Vec4(f32 x = 0, f32 y = 0, f32 z = 0, f32 w = 0) : mElements{ x, y, z, w } {}
	__device__ __host__ Vec4(const Vec4& v) : mElements{v[0], v[1], v[2], v[3]} {}

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

	__device__ __host__ Vec4 operator-() const;

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

private:
	f32 mElements[4];
};