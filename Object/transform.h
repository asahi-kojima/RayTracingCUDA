#pragma once
#include "Math/matrix.h"

struct Transform
{
public:
    __device__ __host__ Transform();

	__device__ __host__ void setScaling(f32 scale_x, f32 scale_y, f32 scale_z);
	__device__ __host__ void setRotationAngle(f32 angle_x, f32 angle_y, f32 angle_z);
	__device__ __host__ void setTranslation(const Vec3& t);

    __device__ __host__ const Mat4& getTransformMatrix();
    __device__ __host__ const Mat4& getInvTransformMatrix();
    __device__ __host__ const Mat4& getInvTransposeTransformMatrix();

	__device__ __host__ const Vec3& getTranslation() const;

	__device__ __host__ static Transform translation(const Vec3& v);

private:
	__device__ __host__ void calcTransformMatrix();
	__device__ __host__ void calcInverseTransformMatrix();
	__device__ __host__ void calcInverseTransposeTransformMatrix();

	bool mIsDirty;
	
	f32 mScaling[3];
	f32 mRotation[3];
	Vec3 mTranslation;

    Mat4 mTransformMatrix;
	Mat4 mInvTransformMatrix;
	Mat4 mInvTransposeTransformMatrix;
};