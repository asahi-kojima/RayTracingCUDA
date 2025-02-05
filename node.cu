#include <algorithm>
#include "node.h"


Node::Node(Hittable** hittableList, size_t hittableNum)
	:aabb{}
{
	//リストに１つしかない場合、葉となる。
	if (hittableNum == 1)
	{
		//printf("%d must be 1\n", hittableNum);
		isLeaf = true;
		object = new Object(hittableList[0]);
		//printf("OK\n");

		aabb = object->getAABB();
	}
	//２つ以上オブジェクトがある場合、まだ分割を行う。
	else
	{
		Hittable** lhs_hittables = hittableList;
		lhs_node = new Node(lhs_hittables, hittableNum / 2);

		Hittable** rhs_hittables = hittableList + hittableNum / 2;
		rhs_node = new Node(rhs_hittables, (hittableNum + 1) / 2);

		const AABB lhs_node_aabb = lhs_node->aabb;
		const AABB rhs_node_aabb = rhs_node->aabb;

		aabb = AABB::wraping(lhs_node_aabb, rhs_node_aabb);
	}
}

bool Node::hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{
	//AABBと接触があるか確認する。
	if (!isIntersecting(this->aabb, r, t_min, t_max))
	{
		return false;
	}


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
		bool isHitLhs = lhs_node->hit(r, t_min, t_max, lhsRecord);
		HitRecord rhsRecord;
		bool isHitRhs = rhs_node->hit(r, t_min, t_max, rhsRecord);


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
