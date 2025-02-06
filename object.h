#pragma once
#include <vector>
#include <memory>
#include "vector.h"
#include "hittable.h"
#include "util.h"

//#define testMin(a, b) (a < b ? a : b)
//#define testMax(a, b) (a > b ? a : b)

class AABB : public Hittable
{
public:
	__device__ AABB() : minPos(0, 0, 0), maxPos(0, 0, 0) {}
	__device__ AABB(vec3 minPos, vec3 maxPos)
		: minPos(minPos), maxPos(maxPos) {}

	__device__ const vec3& getMinPos() const { return minPos; }
	__device__ const vec3& getMaxPos() const { return maxPos; }

	__device__ inline static AABB wraping(AABB lhs, AABB rhs)
	{
		vec3 minPos;
		{
			minPos.setX(fminf(lhs.minPos.getX(), rhs.minPos.getX()));
			minPos.setY(fminf(lhs.minPos.getY(), rhs.minPos.getY()));
			minPos.setZ(fminf(lhs.minPos.getZ(), rhs.minPos.getZ()));
		}
		vec3 maxPos;
		{
			maxPos.setX(fmaxf(lhs.maxPos.getX(), rhs.maxPos.getX()));
			maxPos.setY(fmaxf(lhs.maxPos.getY(), rhs.maxPos.getY()));
			maxPos.setZ(fmaxf(lhs.maxPos.getZ(), rhs.maxPos.getZ()));
		}
		return AABB(minPos, maxPos);
	}

	__device__ vec3 getCenterPos() const
	{
		return (minPos + maxPos) / 2.0f;
	}

	__device__ void printSize() const
	{
		printf("maxPos = %f, %f, %f\n", maxPos[0], maxPos[1], maxPos[2]);		
		printf("minPos = %f, %f, %f\n", minPos[0], minPos[1], minPos[2]);		
	}

	__device__  bool isIntersecting(const Ray& ray, f32 t_min, f32 t_max) const;

private:
	__device__ virtual bool hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ virtual AABB calcAABB() override { return *this; }


	vec3 minPos;
	vec3 maxPos;
};

__device__ AABB wraping(AABB lhs, AABB rhs);

class Sphere : public Hittable
{
public:
	__device__ Sphere() = default;
	__device__ Sphere(const vec3& center, const f32 radius, Material* material)
		: center(center), radius(radius), material(material) {}


private:
	__device__ bool hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
	__device__ AABB calcAABB() override;


	vec3 center;
	f32 radius;
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
//	Mesh(std::vector<vec3>&& vertexList, std::vector<f32>&& indexList);
//	Mesh(const std::vector<vec3>& vertexList, const std::vector<f32>& indexList);
//
//private:
//	std::vector<vec3> mVertexList;
//	std::vector<f32> mIndexList;
//
//	bool hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) override;
//	AABB calcAABB() override;
//};