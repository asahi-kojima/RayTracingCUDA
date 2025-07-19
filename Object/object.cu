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



bool Object::isHit(const Ray& ray, const f32 t_min, const f32 t_max, HitRecord& record)
{
    const Mat4& invTransformMat= mTransform.getInvTransformMatrix();
    Ray rayTransformedIntoObjectSpace = ray.transformWith(invTransformMat);

    bool isHitToPrimitive = mPrimitivePtr->isHit(rayTransformedIntoObjectSpace, t_min, t_max, record);

    if (!isHitToPrimitive)
    {
        return false;
    }


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

Vec3 Object::getRandomPointOnSurfaceVisibleFrom(const Vec3& point, bool& isVisible) const
{
    Vec4 position(point, 1);
    const Mat4& invTransformMat= mTransform.getInvTransformMatrix();
    Vec3 localPosition = (invTransformMat * position).extractXYZ();

    return mPrimitivePtr->getRandomPointOnSurfaceVisibleFrom(localPosition, isVisible);
}

void Object::calculateLightSampling(const Vec3& rayOrigin, const Transform& transform, Vec3& samplingPointOnSurface ,bool& isVisibleFromRayOrigin, f32& amplitude) const
{
    Vec4 position(rayOrigin, 1);
    const Mat4& invTransformMat= mTransform.getInvTransformMatrix();
    Vec3 localPosition = (invTransformMat * position).extractXYZ();

    mPrimitivePtr->calculateLightSampling(localPosition, mTransform, samplingPointOnSurface, isVisibleFromRayOrigin, amplitude);

    {
        Vec4 position(samplingPointOnSurface, 1);
        samplingPointOnSurface = (mTransform.getTransformMatrix() * position).extractXYZ();
    }
}
