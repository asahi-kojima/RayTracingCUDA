#pragma once
#include "common.h"
#include "util.h"
#include "hittable.h"
#include "primitive.h"

class BvhNode : public Hittable
{
public:
    BvhNode();
    ~BvhNode();

private:
    BvhNode* mLhsNode;
    BvhNode* mRhsNode;
    // AABB
	__device__ virtual bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB getAABB() = 0;

    AABB mAABB;
};