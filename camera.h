#pragma once
#include <math.h>
#include <cmath>
#include "ray.h"
#include "Math/vector.h"
#include "util.h"

class Camera
{
public:
	__device__ __host__ Camera() = default;
	__device__ __host__ Camera(const Camera&) = default;
	__device__ __host__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, f32 vfov, f32 aspect, f32 aperture = 0, f32 focusDist = 1);

	__device__ Ray getRay(f32 s, f32 t);

	__device__ __host__ Vec3 getEyeOrigin() const { return mEyeOrigin; }
	__device__ __host__ Vec3 getScreenOrigin() const { return mScreenOrigin; }
	__device__ __host__ Vec3 getCameraX() const { return mCameraX; }
	__device__ __host__ Vec3 getCameraY() const { return mCameraY; }
	__device__ __host__ Vec3 getCameraZ() const { return mCameraZ; }
	__device__ __host__ f32 getHorizontalScreenScale() const { return Vec3::dot(horizontal, mCameraX); }
	__device__ __host__ f32 getVerticalScreenScale() const { return Vec3::dot(vertical, mCameraY); }
	__device__ __host__ f32 getFocusDistance() const { return focusDistance; }

private:
	Vec3 mEyeOrigin;
	Vec3 mScreenOrigin;
	Vec3 horizontal;
	Vec3 vertical;
	Vec3 mCameraX, mCameraY, mCameraZ;
	f32 lensRadius;
	f32 focusDistance;

	__device__ Vec3 random_in_unit_disk()
	{
		const f32 theta = RandomGeneratorGPU::uniform_real() * M_2_PI;
		const f32 radius = RandomGeneratorGPU::uniform_real();

		return Vec3(radius * cos(theta) , radius * sin(theta), 0);
	}
};
