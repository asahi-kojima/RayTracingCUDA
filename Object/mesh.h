#pragma once
#include "common.h"
#include "hittable.h"
#include "primitive.h"

// Trianbleメッシュ
class Triangle final : public Primitive
{
public:
	__device__ Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, Material* material, bool isCulling = true)
	: mVertices{v0, v1, v2}, mMaterial(material) ,mNormal(Vec3::cross(v1 - v0, v2 - v0).normalize()), mIsCulling(isCulling)
	{
		const f32 dot0 = Vec3::dot(v1 - v0, v2 - v0);
		if (dot0 < 0)
		{
			const f32 radius = (v1 - v2).length() / 2.0f;
			const Vec3 center = (v1 + v2) / 2.0f;
			mAABB = AABB(center - radius, center + radius);
			return;
		}

		const f32 dot1 = Vec3::dot(v0 - v1, v2 - v1);
		if (dot1 < 0)
		{
			const f32 radius = (v0 - v2).length() / 2.0f;
			const Vec3 center = (v0 + v2) / 2.0f;
			mAABB = AABB(center - radius, center + radius);
			return;
		}

		const f32 dot2 = Vec3::dot(v1 - v2, v0 - v2);
		if (dot2 < 0)
		{
			const f32 radius = (v1 - v0).length() / 2.0f;
			const Vec3 center = (v1 + v0) / 2.0f;
			mAABB = AABB(center - radius, center + radius);
			return;
		}


		const Vec3 c1 = (v1 + v0) / 2.0f;
		const Vec3 c2 = (v2 + v0) / 2.0f;

		const Vec3 n1 = Vec3::cross(v1 - v0, mNormal);
		const Vec3 n2 = Vec3::cross(v2 - v0, mNormal);

		const f32 a = Vec3::dot(n1, n1);
		const f32 b = Vec3::dot(n1, n2);
		const f32 c = -Vec3::dot(n2, n2);
		const f32 e = Vec3::dot(n1, c2- c1);
		const f32 f = Vec3::dot(n2, c2- c1);

		const f32 det = a * c + b * b;
		const f32 t = (c * e + b * f) / det;
		//const f32 s = (-b * e + a * f) / det;

		const Vec3 circumcenter = c1 + t * n1;
		const f32 radius = (circumcenter - v0).length();
		mAABB = AABB(circumcenter - radius, circumcenter + radius);
	}


	
	private:
	__device__ bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const override;
	__device__ __host__ AABB getAABB() override;

	Vec3 mVertices[3];
	Vec3 mNormal;
	Material* mMaterial;
	AABB mAABB;
	bool mIsCulling;
};


class SubMesh : public Hittable
{
public:
    SubMesh();
    ~SubMesh();

    __device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const = 0;
	__device__ __host__ virtual AABB getAABB() = 0;

private:
    Triangle* mTriangleList;
    u32 mShaderId;
};

class Mesh : public Hittable
{
public:
    Mesh();
    ~Mesh();

    __device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const = 0;
	__device__ __host__ virtual AABB getAABB() = 0;

private:
    SubMesh* mSubMeshList;
    u32 mSubMeshNum;
};