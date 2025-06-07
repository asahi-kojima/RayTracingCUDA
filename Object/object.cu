#include "object.h"

Object::Object(Hittable* pritmitivePtr, Material* materialPtr, const Transform& transform)
: mPrimitivePtr(pritmitivePtr)
, mMaterialPtr(materialPtr)
, mTransform(transform)
, mAABB()
, mIsDirty(false)
{
    AABB primitiveAABB = mPrimitivePtr->getAABB();
    const Mat4& transformMat = mTransform.getTransformMatrix();
    mAABB = primitiveAABB.tranformWith(transformMat);
}



bool Object::isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{
    const Mat4& invTransformMat= mTransform.getInvTransformMatrix();
    Ray transformedRay = r.transformWith(invTransformMat);

    bool isHitToPrimitive = mPrimitivePtr->isHit(transformedRay, t_min, t_max, record);

    if (!isHitToPrimitive)
    {
        return false;
    }


    //最適化ポイント：これは衝突点がより近い点があった場合に無駄が出得る。
    //recode内の衝突点を変換する必要がある。
    {
        const Vec4 position(record.position, 1.0f);
        const Mat4& transformMat = mTransform.getTransformMatrix();
        record.position = (transformMat * position).extractXYZ();
    }
    //最適化ポイント：これは衝突点がより近い点があった場合に無駄が出得る。
    //recode内のノーマルを変換する必要がある。
    {
        const Vec4 normal(record.normal, 0.0f);
        const Mat4& invTransposeTransformMat = mTransform.getInvTransposeTransformMatrix();

        record.normal = (invTransposeTransformMat * normal).extractXYZ();
    }

    record.material = mMaterialPtr;

    return true;
}

AABB Object::getAABB()
{
    return mAABB;
}