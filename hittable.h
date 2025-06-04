#pragma once
#include <memory>
#include "material.h"
#include "matrix.h"
#include "transform.h"




class AABB;
class Ray;


struct HitRecord
{
	f32 t;								//���C��������܂ł̃p�����[�^�l
	Vec3 position;							//�ǂ��œ���������
	Vec3 normal;						//�@�������͂ǂ���
	Material* material;	//�ǂ̂悤�ȍގ���
};


class Hittable
{
public:
	__device__ __host__ Hittable()
	: mIsDraw(true)
	, mTransform{} 
	{}

	__device__ virtual bool isHit(const Ray& ray_in, const f32 t_min, const f32 t_max, HitRecord& record) final
	{
		const Ray transformed_ray = transformRayIntoLocal(ray_in);
		return isHitInLocalSpace(transformed_ray, t_min, t_max, record);
	}

	__device__ virtual bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB getAABB() = 0;

	//トランスフォームの値を設定 
	__device__ void setScaling(f32 sx, f32 sy, f32 sz)
	{
		mTransform.setScaling(sx, sy, sz);
	}
	__device__ void setRotationAngle(f32 angle_x, f32 angle_y, f32 angle_z)
	{
		mTransform.setRotationAngle(angle_x, angle_y, angle_z);
	}
	__device__ void setTranslation(const Vec3& t)
	{
		mTransform.setTranslation(t);
	}

	__device__ const Transform& getTransform() {return mTransform;}
	
private:
	//レイをオブジェクトのローカル空間へと移す
	__device__ Ray transformRayIntoLocal(const Ray& original_ray)
	{
		const Vec3& origin = original_ray.origin();
		const Vec3& direction = original_ray.direction();

		const Vec4 originAsVec4(origin, 1);
		const Vec4 directionAsVec4(direction, 0);

		const Mat4& invTransformMat = mTransform.getInvTransformMatrix();

		const Vec4 invTransformedOrigin = invTransformMat * originAsVec4;
		const Vec4 invTransformedDirection = invTransformMat * directionAsVec4;

		return Ray(invTransformedOrigin.extractXYZ(), invTransformedDirection.extractXYZ());
	}

	bool mIsDraw;
	Transform mTransform;
};