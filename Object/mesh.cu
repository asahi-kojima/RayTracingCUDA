#include "mesh.h"
#include "object.h"


bool Triangle::isHit(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{
	const Vec3 p1 = mVertices[1] - mVertices[0];
	const Vec3 p2 = mVertices[2] - mVertices[0];
	const Vec3 v0ToO = ray.origin() - mVertices[0];

	const Vec3 a0 = -ray.direction();
	const Vec3 a1 = p1;
	const Vec3 a2 = p2;

	const Vec3 cross1x2 = Vec3::cross(a1, a2);
	const Vec3 cross2x0 = Vec3::cross(a2, a0);
	const Vec3 cross0x1 = Vec3::cross(a0, a1);

	const f32 det = Vec3::dot(cross1x2, a0);
	if (det == 0.0)
	{
		return false;
	}

	const f32 t = Vec3::dot(cross1x2, v0ToO) / det;
	const f32 alpha = Vec3::dot(cross2x0, v0ToO) / det;
	const f32 beta = Vec3::dot(cross0x1, v0ToO) / det;

	if (!(t > t_min && t < t_max && alpha + beta < 1 && alpha > 0 && beta > 0))
	{
		return false;
	}


	record.t = t;
	record.position = ray.pointAt(t);
	record.normal = mNormal * (mIsCulling ? 1 : (Vec3::dot(ray.direction(), mNormal) < 0) ? 1 : -1);
	record.material = mMaterial;
	return true;
}

AABB Triangle::getAABB()
{
	return mAABB;
}

bool SubMesh::isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{

}




bool Mesh::isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{
    f32 current_t_max = t_max;
    bool isHitWithSomething = false;
    for (u32 i = 0; i < mSubMeshNum; i++)
    {
        HitRecord tmp_record;
        if (mSubMeshList[i].isHit(r, t_min, current_t_max, tmp_record))
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

    return aabb;
}