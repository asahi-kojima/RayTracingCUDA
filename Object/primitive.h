#pragma once
#include <vector>
#include <memory>
#include "Math/vector.h"
#include "hittable.h"
#include "util.h"

using Primitive = Hittable;

class AABB final : public Hittable
{
public:
	__device__ __host__ AABB() : minPos(0, 0, 0), maxPos(0, 0, 0), mCenter(0, 0, 0) {}
	__device__ __host__ AABB(Vec3 minPos, Vec3 maxPos, Material* material = nullptr)
		: minPos(minPos), maxPos(maxPos), mCenter((minPos + maxPos) / 2),  material(material) {}

	__device__ const Vec3& getMinPos() const { return minPos; }
	__device__ const Vec3& getMaxPos() const { return maxPos; }

	__device__ __host__ inline static AABB wraping(const AABB& lhs, const AABB& rhs)
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

	//軽量のヒット判定関数
	__device__  bool isIntersecting(const Ray& ray,  f32 t_min,  f32 t_max) const;

	__device__ __host__ AABB tranformWith(const Mat4& transformMat) const;
	
private:
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ __host__ virtual AABB getAABB() override { return *this; }


	Vec3 minPos;
	Vec3 maxPos;
	Vec3 mCenter;
	Material* material;
};

class Box final: public Hittable
{
public:
	__device__ Box() : mAABB(-Vec3(1,1,1), Vec3(1,1,1)){}

private:
	__device__ bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ __host__ AABB getAABB() override;

	AABB mAABB;
};



class Sphere final: public Hittable
{
public:
	__device__ Sphere() : mAABB(-Vec3(1,1,1), Vec3(1,1,1)){}


private:
	__device__ bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ __host__ AABB getAABB() override;

	AABB mAABB;
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
//	bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
//	AABB getAABB() override;
//};