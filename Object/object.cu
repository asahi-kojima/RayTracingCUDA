#include "object.h"

SurfaceProperty::SurfaceProperty()
: mTransparency(0.0f)
, mReflectance(1.0f)
, mAlbedo(0xFFFFFF)
{
}



void SurfaceProperty::setAlbedo(const Color& albedo)
{
    mAlbedo = albedo;
}

const Color& SurfaceProperty::getAlbedo() const
{
    return mAlbedo;
}




Object::Object(Hittable* pritmitivePtr, Material* materialPtr, const Transform& transform, const SurfaceProperty& surfacePropery)
: mPrimitivePtr(pritmitivePtr)
, mMaterialPtr(materialPtr)
, mTransform(transform)
, mAABB()
, mIsDirty(false)
, mSurfaceProperty(surfacePropery)
{
    mTransform.updateTransformMatrices();

    AABB primitiveAABB = mPrimitivePtr->getAABB();
    const Mat4& transformMat = mTransform.getTransformMatrix();
    mAABB = primitiveAABB.tranformWith(transformMat);
}



bool Object::isHit(const Ray& ray, const f32 t_min, const f32 t_max, HitRecord& record) const
{
    const Mat4& invTransformMat= mTransform.getInvTransformMatrix();
    Ray rayTransformedIntoObjectSpace = ray.transformWith(invTransformMat);

    bool isHitToPrimitive = mPrimitivePtr->isHit(rayTransformedIntoObjectSpace, t_min, t_max, record);

    if (!isHitToPrimitive)
    {
        return false;
    }

    //衝突点などの情報はprimitivePtr->isHitのほうで格納している。
    record.position = ray.pointAt(record.t);
    record.material = mMaterialPtr;
    record.hitObject = this;

    return true;
}

AABB Object::getAABB()
{
    return mAABB;
}

const Transform& Object::getTransform() const
{
    return mTransform;
}

const SurfaceProperty& Object::getSurfaceProperty() const
{
    return mSurfaceProperty;
}

Vec3 Object::getRandomPointOnSurface() const
{
    const Mat4& TransformMat= mTransform.getTransformMatrix();
    Vec4 ramdomPointOnPrimitiveSurface(mPrimitivePtr->getRandomPointOnSurface(), 1);
    return (TransformMat * ramdomPointOnPrimitiveSurface).extractXYZ();
}



f32 Object::getPdfValue(const Vec3& origin, const Vec3& direction) const
{
    //まずオブジェクトとヒットするか確認する。
    Ray ray(origin, direction);
    HitRecord record; 
    if (!isHit(ray, 0, INFINITY, record))
    {
        return 0.0f;
    }

    //ヒットする場合はPDFを計算
    return mPrimitivePtr->calcPdfValue(origin, direction, record.position, mTransform);
}


Vec3 Object::generateRandomDirection(const Vec3& origin) const
{
    const Vec4 randomPointOnPrimitive(mPrimitivePtr->getRandomPointOnSurface(), 1);
    const Mat4& transformMat = mTransform.getTransformMatrix();

    const Vec3 worldRamdomPoint = (transformMat * randomPointOnPrimitive).extractXYZ();

    return (worldRamdomPoint - origin).normalize();
}