#include "mesh.h"
#include "object.h"




bool SubMesh::isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{

}




bool Mesh::isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{
    f32 current_t_max = t_max;
    bool isHitWithSomething = false;
    for (u32 i = 0; i < mSubMeshNum; i++)
    {
        HitRecord tmp_record;
        if (mSubMeshList[i].isHitInLocalSpace(r, t_min, current_t_max, tmp_record))
        {
            current_t_max = tmp_record.t;
            isHitWithSomething = true;
        }
    }

    return isHitWithSomething;
}

AABB Mesh::getAABB()
{
    AABB aabb = mSubMeshList[0].getAABB();
    for (u32 i = 1; i < mSubMeshNum; i++)
    {
        aabb = AABB::wraping(aabb, mSubMeshList[i].getAABB());
    }
    
    const Transform& transform = getTransform();
    aabb = aabb.transformWith(transform)

    return aabb;
}