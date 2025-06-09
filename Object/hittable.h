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
	Object* hitObject;
	Material* material;	//�ǂ̂悤�ȍގ���
	u32 bvhDepth;
};


class Hittable
{
public:
	__device__ __host__ Hittable() = default;
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ __host__ virtual AABB getAABB() = 0;
};


