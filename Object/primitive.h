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
	constexpr static f32 DefaultExtensionRange= 0.5f;
	__device__ __host__ AABB() : mMinPosition(-DefaultExtensionRange, -DefaultExtensionRange, -DefaultExtensionRange), mMaxPosition(DefaultExtensionRange,DefaultExtensionRange,DefaultExtensionRange), mCenter(0, 0, 0) {}
	__device__ __host__ AABB(Vec3 minPos, Vec3 maxPos, Material* material = nullptr)
		: mMinPosition(minPos), mMaxPosition(maxPos), mCenter((minPos + maxPos) / 2){}

	__device__ const Vec3& getMinPos() const { return mMinPosition; }
	__device__ const Vec3& getMaxPos() const { return mMaxPosition; }

	__device__ __host__ static AABB wraping(const AABB& lhs, const AABB& rhs);

	__device__ const Vec3& getCenterPos() const;

	__device__ void print_debug() const
	{
		printf("maxPos = %f, %f, %f\n", mMaxPosition[0], mMaxPosition[1], mMaxPosition[2]);		
		printf("minPos = %f, %f, %f\n", mMinPosition[0], mMinPosition[1], mMinPosition[2]);		
	}

	//軽量のヒット判定関数
	__device__  bool isIntersecting(const Ray& ray,  f32 t_min,  f32 t_max) const;

	__device__ __host__ AABB tranformWith(const Mat4& transformMat) const;
	
private:
	__device__ virtual bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const override;
	__device__ __host__ virtual AABB getAABB() override { return *this; }


	Vec3 mMinPosition;
	Vec3 mMaxPosition;
	Vec3 mCenter;
};

class Box final: public Hittable
{
public:
	constexpr static f32 DefaultExtensionRange= 0.5f;

	__device__ Box() : mAABB(-Vec3(DefaultExtensionRange,DefaultExtensionRange,DefaultExtensionRange), Vec3(DefaultExtensionRange,DefaultExtensionRange,DefaultExtensionRange)){}

private:
	__device__ bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const override;
	__device__ __host__ AABB getAABB() override;

	AABB mAABB;
};



class Sphere final: public Hittable
{
public:
	__device__ Sphere() : mAABB(-Vec3(1,1,1), Vec3(1,1,1)){}


private:
	__device__ bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const override;
	__device__ __host__ AABB getAABB() override;

	AABB mAABB;
};


class Board final: public Hittable
{
public:
	__device__ Board() : mAABB(Vec3(-DefaultEdgeLength,-Thinness,-DefaultEdgeLength), Vec3(DefaultEdgeLength,Thinness,DefaultEdgeLength)){}


private:
	__device__ bool isHit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record) const override;

    __device__ virtual Vec3 getRandomPointOnSurface() const;

	__device__ virtual f32 calcPdfValue(const Vec3& origin, const Vec3& direction, const Vec3& surfacePoint, const Transform& transform) const override;

	__device__ __host__ AABB getAABB() override;

	AABB mAABB;
	constexpr static f32 DefaultEdgeLength = 0.5f;
	constexpr static f32 Thinness = 0.00001f;
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