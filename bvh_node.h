#pragma once
#include "common.h"
#include "util.h"
#include "Object/hittable.h"
#include "Object/primitive.h"

class Object;

class BvhNode : public Hittable
{
public:
    __device__ BvhNode();
    __device__ ~BvhNode();

private:
    BvhNode* mLhsNodeDevicePtr;
    BvhNode* mRhsNodeDevicePtr;
    Object* mObjectDevicePtr;
    // AABB
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) ;
	__device__ __host__ virtual AABB getAABB();


    AABB mAABB;

    bool mIsLeaf;
};