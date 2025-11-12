#include "camera.h"

__device__ __host__ Camera::Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, f32 vfov, f32 aspect, f32 aperture, f32 focusDist)
{
    mLensRadius = aperture / 2;
    mFocusDistance = focusDist;

    f32 theta = vfov * M_PI / 180.0f;
    f32 halfHeight = tan(theta / 2);
    f32 halfWidth = aspect * halfHeight;

    mEyeOrigin = lookFrom;
	mTarget = lookAt;
    mCameraZ = Vec3::normalize(lookFrom - lookAt);
    mCameraX = Vec3::normalize(Vec3::cross(vUp, mCameraZ));
    mCameraY = Vec3::cross(mCameraZ, mCameraX);

    mScreenOrigin = mEyeOrigin - focusDist * mCameraZ - focusDist * halfWidth * mCameraX - focusDist * halfHeight * mCameraY;
    mHorizontalDirection = focusDist * 2 * halfWidth * mCameraX;
    mVerticalDirection = focusDist * 2 * halfHeight * mCameraY;
}

__device__ Ray Camera::getRay(f32 s, f32 t)
{
    //Vec3 rd = lensRadius * random_in_unit_disk();
    //Vec3 offset = mCameraX * rd[0] + mCameraY * rd[1];

	Vec3 offset = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 rayOrigin = mEyeOrigin + offset;

    return Ray(rayOrigin, Vec3::normalize(mScreenOrigin + s * mHorizontalDirection + t * mVerticalDirection - rayOrigin));
}