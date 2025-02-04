#pragma once
#include <memory>
#include "material.h"


struct HitRecord
{
	f32 t;								//レイが当たるまでのパラメータ値
	vec3 pos;							//どこで当たったか
	vec3 normal;						//法線方向はどこか
	Material* material;	//どのような材質か
};


class AABB;
class Ray;
class Hittable
{
public:
	__device__ virtual bool hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB calcAABB() = 0;
};