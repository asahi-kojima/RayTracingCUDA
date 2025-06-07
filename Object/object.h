#pragma once
#include "common.h"
#include "Math/matrix.h"
#include "primitive.h"

class Object : public Hittable
{
public:
    __device__ Object(Primitive* pritmitivePtr, Material* materialPtr, const Transform& transform = Transform());
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record);
	__device__ __host__ virtual AABB getAABB();

private:
    //参照するプリミティブメッシュの名前とポインタ
    //const char* mPrimitiveName;
    Primitive* mPrimitivePtr;

    //参照するマテリアルの名前とポインタ
    //const char* mMaterialName;
    Material* mMaterialPtr;

    //トランスフォーム行列
    Transform mTransform;

    // AABB
    AABB mAABB;

    // トランスフォームの変化を検知する
    bool mIsDirty;
};