#pragma once
#include "common.h"
#include "util.h"
#include "Object/hittable.h"
#include "Object/primitive.h"

class Object;

class BvhNode : public Hittable
{
public:
    __device__ BvhNode() = default;
    __device__ ~BvhNode() = default;

    __device__ BvhNode(BvhNode* lhs, BvhNode* rhs, const AABB& aabb)
    : mLhsNodeDevicePtr(lhs)
    , mRhsNodeDevicePtr(rhs)
    , mObjectDevicePtr(nullptr)
    , mAABB(aabb)
    , mIsLeaf(false)
    {
    }

    __device__ BvhNode(Object* objectPtr, const AABB& aabb)
    : mLhsNodeDevicePtr(nullptr)
    , mRhsNodeDevicePtr(nullptr)
    , mObjectDevicePtr(objectPtr)
    , mAABB(aabb)
    , mIsLeaf(true)
    {
    }
    
    __device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) ;
    __device__ __host__ virtual AABB getAABB();
    
private:
    BvhNode* mLhsNodeDevicePtr;
    BvhNode* mRhsNodeDevicePtr;
    Object* mObjectDevicePtr;
    
    // AABB
    
    
    AABB mAABB;
    
    bool mIsLeaf;
};