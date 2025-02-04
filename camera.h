#pragma once
#include <math.h>
#include <cmath>
#include "ray.h"
#include "vector.h"
#include "util.h"

class CameraCore/* : public Managed*/
{
	friend class Camera;
public:
	__device__ __host__ CameraCore() = default;
	__device__ __host__ CameraCore(vec3 lookFrom, vec3 lookAt, vec3 vUp, f32 vfov, f32 aspect, f32 aperture = 0, f32 focusDist = 1)
	{
		lensRadius = aperture / 2;
		focusDistance = focusDist;

		f32 theta = vfov * M_PI / 180.0f;
		f32 halfHeight = tan(theta / 2);
		f32 halfWidth = aspect * halfHeight;

		mEyeOrigin = lookFrom;
		mCameraZ = normalize(lookFrom - lookAt);//z
		mCameraX = normalize(cross(vUp, mCameraZ));//x
		mCameraY = cross(mCameraZ, mCameraX);//y

		mScreenOrigin = mEyeOrigin - focusDist * mCameraZ - focusDist * halfWidth * mCameraX - focusDist * halfHeight * mCameraY;
		horizontal = focusDist * 2 * halfWidth * mCameraX;
		vertical = focusDist * 2 * halfHeight * mCameraY;
	}


	__device__ __host__ Ray getRay(f32 s, f32 t)
	{
		vec3 offset;
		//if (lensRadius > 1e-5)
		//{
		//	vec3 rd = lensRadius * random_in_unit_disk();
		//	offset = mCameraX * rd[0] + mCameraY * rd[1];
		//}
		//else
		{
			offset = vec3::zero();
		}
		vec3 rayOrigin = mEyeOrigin + offset;
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
	vec3 mScreenOrigin;//���������_
	vec3 horizontal;
	vec3 vertical;
	vec3 mCameraX, mCameraY, mCameraZ;
	f32 lensRadius;
	f32 focusDistance;


	///// <summary>
	///// �w���p�[�֐�
	///// </summary>
	///// <returns></returns>
	//static vec3 random_in_unit_disk()
	//{
	//	vec3 p;
	//	do
	//	{
	//		p = vec3(RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real(), 0);
	//	} while (dot(p, p) >= 1.0f);

	//	return p;
	//}
};


class Camera
{
public:
	Camera(vec3 lookFrom, vec3 lookAt, vec3 vUp, f32 vfov, f32 aspect, f32 aperture = 0, f32 focusDist = 1)
	{
		cudaMallocManaged(&pCore, sizeof(Camera));


		pCore->lensRadius = aperture / 2;
		pCore->focusDistance = focusDist;

		f32 theta = vfov * M_PI / 180.0f;
		f32 halfHeight = tan(theta / 2);
		f32 halfWidth = aspect * halfHeight;

		pCore->mEyeOrigin = lookFrom;
		pCore->mCameraZ = normalize(lookFrom - lookAt);//z
		pCore->mCameraX = normalize(cross(vUp, pCore->mCameraZ));//x
		pCore->mCameraY = cross(pCore->mCameraZ, pCore->mCameraX);//y

		pCore->mScreenOrigin = pCore->mEyeOrigin - focusDist * pCore->mCameraZ - focusDist * halfWidth * pCore->mCameraX - focusDist * halfHeight * pCore->mCameraY;
		pCore->horizontal = focusDist * 2 * halfWidth * pCore->mCameraX;
		pCore->vertical = focusDist * 2 * halfHeight *pCore->mCameraY;
	}


	CameraCore* pCore;
};