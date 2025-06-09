#include "bvh_node.h"
#include "Object/object.h"

bool BvhNode::isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{

    if (mIsLeaf)
    {
        return mObjectDevicePtr->isHit(r, t_min, t_max, record);
    }
    
    if (!mAABB.isIntersecting(r, t_min, t_max))
    {
        return false;
    }
    record.bvhDepth = record.bvhDepth + 1;
    
    bool isHitLhs = mLhsNodeDevicePtr->isHit(r, t_min, t_max, record);
    
    bool isHitRhs = mRhsNodeDevicePtr->isHit(r, t_min, t_max, record);

    return (isHitLhs || isHitRhs);
}

AABB BvhNode::getAABB()
{
    return mAABB;
}