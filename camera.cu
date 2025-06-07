#include "camera.h"

__device__ __host__ Camera::Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, f32 vfov, f32 aspect, f32 aperture, f32 focusDist)
{
    lensRadius = aperture / 2;
    focusDistance = focusDist;

    f32 theta = vfov * M_PI / 180.0f;
    f32 halfHeight = tan(theta / 2);
    f32 halfWidth = aspect * halfHeight;

    mEyeOrigin = lookFrom;
    mCameraZ = Vec3::normalize(lookFrom - lookAt);	// z
    mCameraX = Vec3::normalize(Vec3::cross(vUp, mCameraZ)); // x
    mCameraY = Vec3::cross(mCameraZ, mCameraX);		// y

    mScreenOrigin = mEyeOrigin - focusDist * mCameraZ - focusDist * halfWidth * mCameraX - focusDist * halfHeight * mCameraY;
    horizontal = focusDist * 2 * halfWidth * mCameraX;
    vertical = focusDist * 2 * halfHeight * mCameraY;
}

__device__ Ray Camera::getRay(f32 s, f32 t)
{
    Vec3 rd = lensRadius * random_in_unit_disk();
    Vec3 offset = mCameraX * rd[0] + mCameraY * rd[1];

    Vec3 rayOrigin = mEyeOrigin + offset;
 
    return Ray(rayOrigin, Vec3::normalize(mScreenOrigin + s * horizontal + t * vertical - rayOrigin));
}