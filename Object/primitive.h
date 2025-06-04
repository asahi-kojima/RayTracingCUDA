#pragma once
#include <vector>
#include <memory>
#include "vector.h"
#include "hittable.h"
#include "util.h"


class AABB final : public Hittable
{
public:
	__device__ AABB() : minPos(0, 0, 0), maxPos(0, 0, 0), mCenter(0, 0, 0) {}
	__device__ AABB(Vec3 minPos, Vec3 maxPos, Material* material = nullptr)
		: minPos(minPos), maxPos(maxPos), mCenter((minPos + maxPos) / 2),  material(material) {}

	__device__ const Vec3& getMinPos() const { return minPos; }
	__device__ const Vec3& getMaxPos() const { return maxPos; }

	__device__ inline static AABB wraping(AABB lhs, AABB rhs)
	{
		Vec3 minPos;
		{
			minPos.x() = fminf(lhs.minPos.x(), rhs.minPos.x());
			minPos.y() = fminf(lhs.minPos.y(), rhs.minPos.y());
			minPos.z() = fminf(lhs.minPos.z(), rhs.minPos.z());
		}
		Vec3 maxPos;
		{
			maxPos.x() = fmaxf(lhs.maxPos.x(), rhs.maxPos.x());
			maxPos.y() = fmaxf(lhs.maxPos.y(), rhs.maxPos.y());
			maxPos.z() = fmaxf(lhs.maxPos.z(), rhs.maxPos.z());
		}
		return AABB(minPos, maxPos);
	}

	__device__ Vec3 getCenterPos() const
	{
		return (minPos + maxPos) / 2.0f;
	}

	__device__ void print_debug() const
	{
		printf("maxPos = %f, %f, %f\n", maxPos[0], maxPos[1], maxPos[2]);		
		printf("minPos = %f, %f, %f\n", minPos[0], minPos[1], minPos[2]);		
	}

	__device__  bool isIntersecting(const Ray& ray,  f32 t_min,  f32 t_max) const;

	__device__ AABB tranformWith(const Transform& transform) const;
	
private:
	__device__ virtual bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ virtual AABB getAABB() override { return *this; }


	Vec3 minPos;
	Vec3 maxPos;
	Vec3 mCenter;
	Material* material;
};

class Triangle final : public Hittable
{
public:
	__device__ Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, Material* material, bool isCulling = true)
	: mVertices{v0, v1, v2}, mMaterial(material) ,mNormal(Vec3::cross(v1 - v0, v2 - v0).normalize()), mIsCulling(isCulling)
	{
		const f32 dot0 = dot(v1 - v0, v2 - v0);
		if (dot0 < 0)
		{
			const f32 radius = (v1 - v2).length() / 2.0f;
			const Vec3 center = (v1 + v2) / 2.0f;
			mAABB = AABB(center - radius, center + radius);
			return;
		}

		const f32 dot1 = dot(v0 - v1, v2 - v1);
		if (dot1 < 0)
		{
			const f32 radius = (v0 - v2).length() / 2.0f;
			const Vec3 center = (v0 + v2) / 2.0f;
			mAABB = AABB(center - radius, center + radius);
			return;
		}

		const f32 dot2 = dot(v1 - v2, v0 - v2);
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

		const f32 a = dot(n1, n1);
		const f32 b = dot(n1, n2);
		const f32 c = -dot(n2, n2);
		const f32 e = dot(n1, c2- c1);
		const f32 f = dot(n2, c2- c1);

		const f32 det = a * c + b * b;
		const f32 t = (c * e + b * f) / det;
		//const f32 s = (-b * e + a * f) / det;

		const Vec3 circumcenter = c1 + t * n1;
		const f32 radius = (circumcenter - v0).length();
		mAABB = AABB(circumcenter - radius, circumcenter + radius);
	}


	
	private:
	__device__ bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ AABB getAABB() override;

	Vec3 mVertices[3];
	Vec3 mNormal;
	Material* mMaterial;
	AABB mAABB;
	bool mIsCulling;
};

class Sphere final: public Hittable
{
public:
	__device__ Sphere() = default;
	__device__ Sphere(){}


private:
	__device__ bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ AABB getAABB() override;

	Material* material;
};


















template <class MaterialKind, typename... Args>
inline __global__ void make_material(Material* p, Args...args)
{
	new (p) MaterialKind(args...);
}

template <class MaterialKind, typename... Args>
inline Material* make_material(Args... args)
{
	Material* pMaterial;
	cudaMalloc(&pMaterial, sizeof(MaterialKind));
	make_material<MaterialKind><<<1,1>>>(pMaterial, args...);
	GPU_ERROR_CHECKER(cudaPeekAtLastError());

	return pMaterial;
}

template <class T, typename...Args>
inline __global__ void make_object(T* p, Args...args)
{
	new (p) T(args...);
}

template <class T, typename...Args>
inline Hittable* make_object(Args...args)
{
	T* p;
	CHECK(cudaMalloc(&p, sizeof(T)));
	make_object<T> << <1, 1 >> > (p, args...);

	CHECK(cudaDeviceSynchronize());
	GPU_ERROR_CHECKER(cudaPeekAtLastError());


	return reinterpret_cast<Hittable*>(p);;
}





















//class Mesh : public Hittable
//{
//public:
//	Mesh(std::vector<Vec3>&& vertexList, std::vector<f32>&& indexList);
//	Mesh(const std::vector<Vec3>& vertexList, const std::vector<f32>& indexList);
//
//private:
//	std::vector<Vec3> mVertexList;
//	std::vector<f32> mIndexList;
//
//	bool isHitInLocalSpace(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
//	AABB getAABB() override;
//};