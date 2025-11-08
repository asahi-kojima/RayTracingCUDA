#pragma once
#include "common.h"
#include "vector.h"


class AABB
{
public:
	AABB(const Vec3& minPosition, const Vec3& maxPosition)
		: mMinPosition(minPosition)
		, mMaxPosition(maxPosition)
	{}

private:
	Vec3 mMinPosition;
	Vec3 mMaxPosition;
};