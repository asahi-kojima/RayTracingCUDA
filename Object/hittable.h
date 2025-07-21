#pragma once
#include <memory>
#include "material.h"
#include "Math/matrix.h"
#include "transform.h"




class AABB;
class Ray;
class Object;

struct HitRecord
{
	f32 t;								//���C��������܂ł̃p�����[�^�l
	Vec3 position;							//�ǂ��œ���������
	Vec3 normal;						//�@�������͂ǂ���
	const Object* hitObject;
	Material* material;	//�ǂ̂悤�ȍގ���
};


class Hittable
{
public:
	__device__ __host__ Hittable() = default;
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const = 0;
	__device__ __host__ virtual AABB getAABB() = 0;

	__device__ virtual Vec3 getRandomPointOnSurface() const {return Vec3(0,0,0);}
	
	__device__ virtual f32 calcPdfValue(const Vec3& origin, const Vec3& direction, const Vec3& surfacePoint, const Transform& transform) const {return 0.0f;}
};


