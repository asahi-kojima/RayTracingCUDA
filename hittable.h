#pragma once
#include <memory>
#include "material.h"
#include "matrix.h"

struct HitRecord
{
	f32 t;								//���C��������܂ł̃p�����[�^�l
	Vec3 pos;							//�ǂ��œ���������
	Vec3 normal;						//�@�������͂ǂ���
	Material* material;	//�ǂ̂悤�ȍގ���
};




class AABB;
class Ray;

class Transfrom
{
public:
	void calcInverseTransform()
	{
		Mat4 inv_T = Mat4::generateTransform(-mTranslation);
		Mat4 inv_R = Mat4::generateRotation(-mRotation[0], -mRotation[1], -mRotation[2]);
		Mat4 inv_S = Mat4::generateScale(1.0f / mScaling[0], 1.0f / mScaling[1], 1.0f / mScaling[2]);

		mInvTransform = inv_S * inv_R * inv_T;
	}

	bool mIsDirty;
	
	f32 mScaling[3];
	f32 mRotation[3];
	Vec3 mTranslation;

	Mat4 mInvTransform;
};

class Hittable
{
public:
	__device__ __host__ Hittable()
	: mIsDraw(true)
	, mIsDirty(true)
	, mScaling{1.0f, 1.0f, 1.0f}
	, mRotation{0.0f, 0.0f, 0.0f}
	, mTranslation{0, 0, 0} {}

	__device__ bool isHit(const Ray& ray_in, const f32 t_min, const f32 t_max, HitRecord& record)
	{
		if (mIsDirty)
		{
			calcInverseTransform();
		}

		const Ray transformed_ray = transformRayIntoLocal(ray_in);
		return hit(transformed_ray, t_min, t_max, record);
	}

	__device__ virtual bool hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB calcAABB() = 0;

	//トランスフォームの値を設定 
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

	//レイをオブジェクトのローカル空間へと移す
	__device__ Ray transformRayIntoLocal(const Ray& original_ray)
	{
		const Vec3& origin = original_ray.origin();
		const Vec3& direction = original_ray.direction();

		const Vec4 originAsVec4(origin, 1);
		const Vec4 directionAsVec4(direction, 0);

		const Vec4 invTransformedOrigin = mInvTransform * originAsVec4;
		const Vec4 invTransformedDirection = mInvTransform * directionAsVec4;

		return Ray(invTransformedOrigin.extractXYZ(), invTransformedDirection.extractXYZ());
	}


private:
	void calcInverseTransform()
	{
		Mat4 inv_T = Mat4::generateTransform(-mTranslation);
		Mat4 inv_R = Mat4::generateRotation(-mRotation[0], -mRotation[1], -mRotation[2]);
		Mat4 inv_S = Mat4::generateScale(1.0f / mScaling[0], 1.0f / mScaling[1], 1.0f / mScaling[2]);

		mInvTransform = inv_S * inv_R * inv_T;
	}

	bool mIsDraw;

	bool mIsDirty;
	
	f32 mScaling[3];
	f32 mRotation[3];
	Vec3 mTranslation;

	Mat4 mInvTransform;
};