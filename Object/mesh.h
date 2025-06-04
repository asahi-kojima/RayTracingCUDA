#pragma once
#include "common.h"
#include "hittable.h"

// Trianbleメッシュ
class Triangle;

class SubMesh : public Hittable
{
public:
    SubMesh();
    ~SubMesh();

    __device__ virtual bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB getAABB() = 0;

private:
    Triangle* mTriangleList;
    u32 mShaderId;
};

class Mesh : public Hittable
{
public:
    Mesh();
    ~Mesh();

    __device__ virtual bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) = 0;
	__device__ virtual AABB getAABB() = 0;

private:
    SubMesh* mSubMeshList;
    u32 mSubMeshNum;
};