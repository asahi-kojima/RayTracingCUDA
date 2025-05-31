#pragma once
#include "vector.h"

class Ray
{
public:
	__device__ __host__  Ray() = default;
	__device__ __host__  Ray(const Ray& ray) : mOrigin(ray.mOrigin), mDirection(ray.mDirection) {}
	__device__ __host__  Ray(const Vec3& a, const Vec3& b) : mOrigin(a), mDirection(b) {}

	__device__ __host__  const Vec3& origin() const { return mOrigin; }
	__device__ __host__  Vec3& origin() { return mOrigin; }
	__device__ __host__  const Vec3& direction() const { return mDirection; }
	__device__ __host__  Vec3& direction() { return mDirection; }

	__device__ __host__  Vec3 pointAt(const f32 t) const { return mOrigin + mDirection * t; }

	__device__ void print_debug() const
	{
		mOrigin.print_debug();
		mDirection.print_debug();
	}

private:
	Vec3 mOrigin;
	Vec3 mDirection;
};