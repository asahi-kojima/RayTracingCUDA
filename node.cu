#include <algorithm>
#include "node.h"

Node::Node(Hittable **hittableList, u32 *newOrderedIndexList, u32 start, u32 end)
	: aabb{}
{
	// ���X�g�ɂP�����Ȃ��ꍇ�A�t�ƂȂ�B
	if (end - start == 1)
	{
		isLeaf = true;
		object = new Object(hittableList[newOrderedIndexList[start]]);
		aabb = object->getAABB();
	}
	// �Q�ȏ�I�u�W�F�N�g������ꍇ�A�܂��������s���B
	else
	{
		lhs_node = new Node(hittableList, newOrderedIndexList, start, start + (end - start) / 2);
		rhs_node = new Node(hittableList, newOrderedIndexList, start + (end - start) / 2, end);

		const AABB lhs_node_aabb = lhs_node->aabb;
		const AABB rhs_node_aabb = rhs_node->aabb;

		aabb = AABB::wraping(lhs_node_aabb, rhs_node_aabb);
	}
}

bool Node::hit(const Ray &r, const f32 t_min, const f32 t_max, HitRecord &record, u32 &bvh_depth) const
{

	// �ڐG������΁A���̓����Ƃ��������Ă���\��������̂ŁA
	// �����̃m�[�h�ɃA�N�Z�X���ɂ����B
	if (isLeaf)
	{
		Hittable *pObject = object->getObject();
		return pObject->hit(r, t_min, t_max, record);
	}
	else
	{
		// AABB�ƐڐG�����邩�m�F����B
		if (!aabb.isIntersecting(r, t_min, t_max))
		{
			return false;
		}
		bvh_depth++;
		
		f32 current_tmax = t_max;

		HitRecord lhsRecord;
		bool isHitLhs = lhs_node->hit(r, t_min, current_tmax, lhsRecord, bvh_depth);
		if (isHitLhs)
		{
			current_tmax = lhsRecord.t;
			record = lhsRecord;
		}

		HitRecord rhsRecord;
		bool isHitRhs = rhs_node->hit(r, t_min, current_tmax, rhsRecord, bvh_depth);
		if (isHitRhs)
		{
			current_tmax = rhsRecord.t;
			record = rhsRecord;
		}

		return (isHitLhs || isHitRhs);
	}
}

Object::Object(Hittable *hittableObject)
	: mAABB(), mGeometry(nullptr)
{
	mGeometry = hittableObject;
	mAABB = mGeometry->calcAABB();
}
