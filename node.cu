#include <algorithm>
#include "node.h"


Node::Node(Hittable** hittableList, u32 *newOrderedIndexList, u32 start, u32 end)
	:aabb{}
{
	//リストに１つしかない場合、葉となる。
	if (end - start == 1)
	{
		isLeaf = true;
		object = new Object(hittableList[newOrderedIndexList[start]]);
		aabb = object->getAABB();
	}
	//２つ以上オブジェクトがある場合、まだ分割を行う。
	else
	{
		lhs_node = new Node(hittableList, newOrderedIndexList, start, start + (end - start) / 2);
		rhs_node = new Node(hittableList, newOrderedIndexList, start + (end - start) / 2, end);

		const AABB lhs_node_aabb = lhs_node->aabb;
		const AABB rhs_node_aabb = rhs_node->aabb;

		aabb = AABB::wraping(lhs_node_aabb, rhs_node_aabb);
	}
}

bool Node::hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record, u32& bvh_depth) const
{
	//AABBと接触があるか確認する。
	if (!aabb.isIntersecting(r, t_min, t_max))
	{
		return false;
	}

	bvh_depth++;

	//接触があれば、その内部とも交差している可能性があるので、
	//内部のノードにアクセスしにいく。
	if (isLeaf)
	{
		Hittable* pObject = object->getObject();
		return pObject->hit(r, t_min, t_max, record);
	}
	else
	{
		HitRecord lhsRecord;
		bool isHitLhs = lhs_node->hit(r, t_min, t_max, lhsRecord, bvh_depth);
		HitRecord rhsRecord;
		bool isHitRhs = rhs_node->hit(r, t_min, t_max, rhsRecord, bvh_depth);


		if (isHitLhs && isHitRhs)
		{
			if (lhsRecord.t < rhsRecord.t)
			{
				record = lhsRecord;
			}
			else
			{
				record = rhsRecord;
			}
		}
		else if (isHitLhs)
		{
			record = lhsRecord;
		}
		else if (isHitRhs)
		{
			record = rhsRecord;
		}
		else
		{
			return false;
		}


		return true;
	}
}



Object::Object(Hittable* hittableObject)
	:mAABB(), mGeometry(nullptr)
{
	mGeometry = hittableObject;
	mAABB = mGeometry->calcAABB();
}
