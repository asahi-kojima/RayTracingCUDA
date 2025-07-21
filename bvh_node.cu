#include "bvh_node.h"
#include "Object/object.h"

bool BvhNode::isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const
{
    if (mIsLeaf)
    {
        return mObjectDevicePtr->isHit(r, t_min, t_max, record);
    }
    
    if (!mAABB.isIntersecting(r, t_min, t_max))
    {
        return false;
    }

    f32 currentTMax = t_max;
    HitRecord recordLhs, recordRhs;
    bool isHitLhs = mLhsNodeDevicePtr->isHit(r, t_min, currentTMax, recordLhs);
    if (isHitLhs)
    {
        currentTMax = recordLhs.t;
        record = recordLhs;
    }
    
    bool isHitRhs = mRhsNodeDevicePtr->isHit(r, t_min, currentTMax, recordRhs);
    if (isHitRhs)
    {
        record = recordRhs;
    }
    

    return (isHitLhs || isHitRhs);
}

AABB BvhNode::getAABB()
{
    return mAABB;
}