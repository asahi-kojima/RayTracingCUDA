#pragma once
#include "vector.h"
#include "matrix.h"

class Ray
{
public:
	__device__ __host__ Ray() = default;
	__device__ __host__ Ray(const Ray& ray)
		: mOrigin(ray.mOrigin)
		, mDirection(ray.mDirection)
		, mTmin(ray.mTmin)
		, mTmax(ray.mTmax)
	{
	}
	__device__ __host__ Ray(const Vec3& origin, const Vec3& direction, const f32 tmin = (1e-7), const f32 tmax = std::numeric_limits<f32>::max())
		: mOrigin(origin)
		, mDirection(direction)
		, mTmin(tmin)
		, mTmax(tmax)
	{ }

	__device__ __host__ const Vec3& origin() const { return mOrigin; }
	__device__ __host__ const Vec3& direction() const { return mDirection; }
	__device__ __host__ f32 tmin() const { return mTmin; }
	__device__ __host__ f32& tmin() { return mTmin; }
	__device__ __host__ f32 tmax() const { return mTmax; }
	__device__ __host__ f32& tmax() { return mTmax; }

	__device__ __host__ Vec3 pointAt(const f32 t) const { return mOrigin + mDirection * t; }
	__device__ __host__ Ray transformWith(const Mat4& mat) const
	{
		const Vec4 origin(mOrigin, 1);
		const Vec4 direction(mDirection, 0);
		
		const Vec4 transformedOrigin = mat * origin;
		const Vec4 transformedDirectin = mat * direction;

		return Ray(transformedOrigin.extractXYZ(), transformedDirectin.extractXYZ());
	}

private:
	Vec3 mOrigin;
	Vec3 mDirection;
	f32 mTmin;
	f32 mTmax;
};