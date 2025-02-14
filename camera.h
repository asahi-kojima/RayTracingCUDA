#pragma once
#include <math.h>
#include <cmath>
#include "ray.h"
#include "vector.h"
#include "util.h"

class Camera
{
	friend class Camera;

public:
	__device__ __host__ Camera() = default;
	__device__ __host__ Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, f32 vfov, f32 aspect, f32 aperture = 0, f32 focusDist = 1)
	{
		lensRadius = aperture / 2;
		focusDistance = focusDist;

		f32 theta = vfov * M_PI / 180.0f;
		f32 halfHeight = tan(theta / 2);
		f32 halfWidth = aspect * halfHeight;

		mEyeOrigin = lookFrom;
		mCameraZ = normalize(lookFrom - lookAt);	// z
		mCameraX = normalize(cross(vUp, mCameraZ)); // x
		mCameraY = cross(mCameraZ, mCameraX);		// y

		mScreenOrigin = mEyeOrigin - focusDist * mCameraZ - focusDist * halfWidth * mCameraX - focusDist * halfHeight * mCameraY;
		horizontal = focusDist * 2 * halfWidth * mCameraX;
		vertical = focusDist * 2 * halfHeight * mCameraY;
	}

	__device__ Ray getRay(f32 s, f32 t)
	{
		// vec3 rd = lensRadius * random_in_unit_disk();
		// vec3 offset = mCameraX * rd[0] + mCameraY * rd[1];

		vec3 rayOrigin = mEyeOrigin;// + offset;

		return Ray(rayOrigin, normalize(mScreenOrigin + s * horizontal + t * vertical - rayOrigin));
	}

	__device__ __host__ vec3 getEyeOrigin() const { return mEyeOrigin; }
	__device__ __host__ vec3 getScreenOrigin() const { return mScreenOrigin; }
	__device__ __host__ vec3 getCameraX() const { return mCameraX; }
	__device__ __host__ vec3 getCameraY() const { return mCameraY; }
	__device__ __host__ vec3 getCameraZ() const { return mCameraZ; }
	__device__ __host__ f32 getHorizontalScreenScale() const { return dot(horizontal, mCameraX); }
	__device__ __host__ f32 getVerticalScreenScale() const { return dot(vertical, mCameraY); }
	__device__ __host__ f32 getFocusDistance() const { return focusDistance; }

private:
	vec3 mEyeOrigin;
	vec3 mScreenOrigin;
	vec3 horizontal;
	vec3 vertical;
	vec3 mCameraX, mCameraY, mCameraZ;
	f32 lensRadius;
	f32 focusDistance;

	__device__ vec3 random_in_unit_disk()
	{
		const f32 theta = RandomGeneratorGPU::uniform_real() * M_2_PI;
		const f32 radius = RandomGeneratorGPU::uniform_real();

		return vec3(radius * cos(theta) , radius * sin(theta), 0);
	}
};
