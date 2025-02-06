#pragma once
#include <cassert>
#include "common.h"

class vec3
{
public:
	__device__ __host__ vec3() = default;
	__device__ __host__ vec3(f32 x, f32 y, f32 z) : mElements{ x, y, z } {}
	__device__ __host__ vec3(const vec3& v) : mElements{v[0], v[1], v[2]} {}

	__device__ __host__  f32 getX() const { return mElements[0]; }
	__device__ __host__  f32 getY() const { return mElements[1]; }
	__device__ __host__  f32 getZ() const { return mElements[2]; }
	__device__ __host__  f32& getX() { return mElements[0]; }
	__device__ __host__  f32& getY() { return mElements[1]; }
	__device__ __host__  f32& getZ() { return mElements[2]; }
	__device__ __host__  void setX(f32 value) { mElements[0] = value; }
	__device__ __host__  void setY(f32 value) { mElements[1] = value; }
	__device__ __host__  void setZ(f32 value) { mElements[2] = value; }

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

	__device__ __host__ vec3 normalize();
	__device__ __host__ f32 length() const;
	__device__ __host__ static f32 length(const vec3&);
	__device__ __host__ f32 lengthSquared() const;
	__device__ __host__ static f32 lengthSquared(const vec3&);
	__device__ __host__ static vec3 cross(const vec3&, const vec3&);

	__device__ __host__ static inline vec3 zero() { return vec3(0.0f, 0.0f, 0.0f); };
	__device__ __host__ static inline vec3 one() { return vec3(1.0f, 1.0f, 1.0f); };

	__device__ __host__ void print() const;

private:
	f32 mElements[3];
};


__device__ __host__ vec3 operator*(const f32 value, const vec3& v);
__device__ __host__ vec3 normalize(const vec3& v);
__device__ __host__ vec3 cross(const vec3& v0, const vec3& v1);
__device__ __host__ f32 dot(const vec3& v0, const vec3& v1);
__device__ __host__ vec3 reflect(const vec3& v, const vec3& n);

__device__ vec3 random_in_unit_sphere();
