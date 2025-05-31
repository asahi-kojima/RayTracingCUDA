#pragma once
#include "matrix.h"

struct Transform
{
public:
	__device__ void setScaling(f32 scale_x, f32 scale_y, f32 scale_z)
	{
#ifdef DEBUG
		if (scale_x == 0.0f || scale_y == 0.0f || scale_z == 0.0f)
		{
			printf("Error : scale value is 0!\n");
		}
#endif
		mScaling[0] = scale_x;
		mScaling[1] = scale_y;
		mScaling[2] = scale_z;
		mIsDirty = true;
	}
	__device__ void setRotationAngle(f32 angle_x, f32 angle_y, f32 angle_z)
	{
		mRotation[0] = angle_x;
		mRotation[1] = angle_y;
		mRotation[2] = angle_z;
		mIsDirty = true;
	}
	__device__ void setTranslation(const Vec3& t)
	{
		mTranslation = t;
		mIsDirty = true;
	}

    __device__ Mat4 getTransformMatrix()
    {
        if (mIsDirty)
        {
            calcTransformMatrix();
            calcInverseTransformMatrix();
            mIsDirty = false;
        }

        return mTransformMatrix;
    }
    __device__ Mat4 getInvTransformMatrix()
    {
        if (mIsDirty)
        {
            calcTransformMatrix();
            calcInverseTransformMatrix();
            mIsDirty = false;
        }

        return mInvTransformMatrix;
    }

private:
	__device__ void calcTransformMatrix()
	{
		Mat4 S = Mat4::generateScale(mScaling[0], mScaling[1], mScaling[2]);
		Mat4 R = Mat4::generateRotation(mRotation[0], mRotation[1], mRotation[2]);
		Mat4 T = Mat4::generateTransform(mTranslation);

		mTransformMatrix = T * R * S;
	}
	__device__ void calcInverseTransformMatrix()
	{
		Mat4 inv_T = Mat4::generateTransform(-mTranslation);
		Mat4 inv_R = Mat4::generateRotation(-mRotation[0], -mRotation[1], -mRotation[2]);
		Mat4 inv_S = Mat4::generateScale(1.0f / mScaling[0], 1.0f / mScaling[1], 1.0f / mScaling[2]);

		mInvTransformMatrix = inv_S * inv_R * inv_T;
	}

	bool mIsDirty;
	
	f32 mScaling[3];
	f32 mRotation[3];
	Vec3 mTranslation;

    Mat4 mTransformMatrix;
	Mat4 mInvTransformMatrix;
};