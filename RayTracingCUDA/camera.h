#pragma once
#include "ray.h"
#include "vector.h"
#include "util.h"



class Camera
{
public:
	__device__ __host__ Camera() = default;
	__device__ __host__ Camera(const Camera&) = default;
	__device__ __host__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, f32 vfov, f32 aspect, f32 aperture = 0, f32 focusDist = 1);

	__device__ Ray getRay(f32 s, f32 t);

	__device__ __host__ Vec3 getEyeOrigin() const { return mEyeOrigin; }
	__device__ __host__ Vec3 getTarget() const { return mTarget; }
	__device__ __host__ Vec3 getScreenOrigin() const { return mScreenOrigin; }
	__device__ __host__ Vec3 getCameraX() const { return mCameraX; }
	__device__ __host__ Vec3 getCameraY() const { return mCameraY; }
	__device__ __host__ Vec3 getCameraZ() const { return mCameraZ; }
	__device__ __host__ f32 getHorizontalScreenScale() const { return Vec3::dot(mHorizontalDirection, mCameraX); }
	__device__ __host__ f32 getVerticalScreenScale() const { return Vec3::dot(mVerticalDirection, mCameraY); }
	__device__ __host__ f32 getFocusDistance() const { return mFocusDistance; }

private:
	Vec3 mEyeOrigin;
	Vec3 mTarget;
	Vec3 mScreenOrigin;
	Vec3 mHorizontalDirection;
	Vec3 mVerticalDirection;
	Vec3 mCameraX, mCameraY, mCameraZ;
	f32 mLensRadius;
	f32 mFocusDistance;

	__device__ Vec3 randomInUnitDisk()
	{
		//const f32 theta = RandomGeneratorGPU::uniform_real() * M_2_PI;
		//const f32 radius = RandomGeneratorGPU::uniform_real();

		//return Vec3(radius * cos(theta), radius * sin(theta), 0);

		return Vec3::unitX() * 0.1f;
	}
};


class CameraController
{
public:
	CameraController()
		: mOrigin(Vec3::zero())
		, mTarget(Vec3::unitZ())
		, mUp(Vec3::unitY())
		, mFovY(45.0f)
		, mAspect(1.0f)
		, mAperture(0.0f)
		, mFocusDistance(1.0f)
	{
	}

	CameraController(const Vec3& origin, const Vec3& target, const Vec3& up, f32 fovY, f32 aspect, f32 aperture = 0.0f, f32 focusDistance = 1.0f)
		: mOrigin(origin)
		, mTarget(target)
		, mUp(up)
		, mFovY(fovY)
		, mAspect(aspect)
		, mAperture(aperture)
		, mFocusDistance(focusDistance)
	{
	}

	void updateCamera(Camera& camera)
	{

	}

	void setOrigin(const Vec3& origin) { mOrigin = origin; }
	void setTarget(const Vec3& target) { mTarget = target; }
	void setUp(const Vec3& up) { mUp = up; }
	void setFovY(f32 fovY) { mFovY = fovY; }
	void setAspect(f32 aspect) { mAspect = aspect; }
	void setAperture(f32 aperture) { mAperture = aperture; }
	void setFocusDistance(f32 focusDistance) { mFocusDistance = focusDistance; }

	void moveOrigin(const Vec3& delta) { mOrigin += delta; }
	void moveTarget(const Vec3& delta) { mTarget += delta; }
	void moveTowardsTarget(f32 delta)
	{
		const Vec3 forward = (mTarget - mOrigin).normalize();
		mOrigin += forward * delta;
	}
	void moveAwayFromTarget(f32 delta)
	{
		const Vec3 forward = (mTarget - mOrigin).normalize();
		mOrigin -= forward * delta;
	}
	void rotateTargetAroundOrigin(const f32 angleAsDegrees, const Vec3& axis)
	{
		const Vec3 direction = mTarget - mOrigin;
		const Mat4 rotationMat = Mat4::generateRotation(axis.normalize(), angleAsDegrees * M_PI / 180.0f);
		const Vec3 rotatedDirection = (rotationMat * Vec4(direction, 0)).extractXYZ();
		mTarget = mOrigin + rotatedDirection;
	}

private:
	Vec3 mOrigin;
	Vec3 mTarget;
	Vec3 mUp;

	f32 mFovY;
	f32 mAspect;
	f32 mAperture; // DOF
	f32 mFocusDistance;
};
