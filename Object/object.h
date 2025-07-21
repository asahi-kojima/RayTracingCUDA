#pragma once
#include "common.h"
#include "Math/matrix.h"
#include "primitive.h"


class SurfaceProperty
{
public:
    __device__ __host__ SurfaceProperty();

    __device__ __host__ void setAlbedo(const Color& albedo);
    __device__ __host__ const Color& getAlbedo() const;

private:
    f32 mTransparency;
    f32 mReflectance;
    Color mAlbedo;
};

class Object : public Hittable
{
public:
    __device__ Object(Primitive* pritmitivePtr, Material* materialPtr, const Transform& transform = Transform(), const SurfaceProperty& surfacePropery = SurfaceProperty());
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const override;
	__device__ __host__ virtual AABB getAABB();
    __device__ __host__ const Transform& getTransform() const;

    __device__ const SurfaceProperty& getSurfaceProperty() const;

    __device__ virtual Vec3 getRandomPointOnSurface() const;

    __device__ virtual f32 getPdfValue(const Vec3& origin, const Vec3& direction) const;
    __device__ virtual Vec3 generateRandomDirection(const Vec3& origin) const;

private:
    // 参照するプリミティブメッシュの名前とポインタ
    Primitive* mPrimitivePtr;

    // 参照するマテリアルの名前とポインタ
    // const char* mMaterialName;
    Material* mMaterialPtr;

    //トランスフォーム行列
    Transform mTransform;

    // AABB
    AABB mAABB;

    // トランスフォームの変化を検知する
    bool mIsDirty;

    // 表面の物性を指定するプロパティ
    SurfaceProperty mSurfaceProperty;
};