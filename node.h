#pragma once
#include "object.h"
#include "ray.h"

class Object
{
public:
	__device__ Object(Hittable* hittableObject);
	~Object() = default;

	__device__ AABB getAABB() const { return mAABB; }
	__device__ Hittable* getObject() const { return mGeometry; }

private:
	AABB mAABB;
	Hittable* mGeometry;
};




struct Node
{
	__device__ Node() = default;
	__device__ Node(const Node&) = default;
	__device__ Node(Node&&) = default;
	//__device__ __host__  Node(std::vector<std::unique_ptr<Object> >&& objectList);
	__device__ Node(Hittable** hittableList,  u32* newOrderedIndexList, u32 start, u32 end);


	__device__ bool hit(const Ray& r,  f32 t_min,  f32 t_max, HitRecord& record) const;


private:
	bool isLeaf = false;
	AABB aabb;
	Object* object = nullptr;

	Node* lhs_node = nullptr;
	Node* rhs_node = nullptr;

};