#pragma once
#include "common.h"
#include "vector.h"


class AABB
{
public:
	AABB() : mMinPosition(Vec3::zero()), mMaxPosition(Vec3::zero()) { assert(0); }
	AABB(const Vec3& minPosition, const Vec3& maxPosition)
		: mMinPosition(minPosition)
		, mMaxPosition(maxPosition)
	{}

private:
	Vec3 mMinPosition;
	Vec3 mMaxPosition;
};