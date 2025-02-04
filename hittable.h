#pragma once
#include <memory>
#include "material.h"


struct HitRecord
{
	f32 t;								//���C��������܂ł̃p�����[�^�l
	vec3 pos;							//�ǂ��œ���������
	vec3 normal;						//�@�������͂ǂ���
	Material* material;	//�ǂ̂悤�ȍގ���
};


class AABB;
class Ray;
class Hittable
{
public:
	__device__ virtual bool hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB calcAABB() = 0;
};