#pragma once
#include "Math/vector.h"
#include "Math/matrix.h"

class Ray
{
public:
	__device__ __host__  Ray() = default;
	__device__ __host__  Ray(const Ray& ray) : mOrigin(ray.mOrigin), mDirection(ray.mDirection) {}
	__device__ __host__  Ray(const Vec3& a, const Vec3& b) : mOrigin(a), mDirection(b) {}

	__device__ __host__  const Vec3& origin() const;
	__device__ __host__  Vec3& origin();
	__device__ __host__  const Vec3& direction() const;
	__device__ __host__  Vec3& direction();

	__device__ __host__  Vec3 pointAt(const f32 t) const;
	
	__device__ Ray transformWith(const Mat4& mat) const ;

	__device__ void print_debug() const;


private:
	Vec3 mOrigin;
	Vec3 mDirection;
};