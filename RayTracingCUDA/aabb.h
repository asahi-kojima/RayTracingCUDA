#pragma once
#include "common.h"
#include "vector.h"

class AABB
{
public:
	AABB() : mMinPosition(Vec3::zero()), mMaxPosition(Vec3::zero()) {}
	AABB(const Vec3& minPosition, const Vec3& maxPosition)
		: mMinPosition(minPosition)
		, mMaxPosition(maxPosition)
	{}

	const Vec3& getMinPosition() const { return mMinPosition; }
	const Vec3& getMaxPosition() const { return mMaxPosition; }

	u32 getMostExtendingAxis() const
	{
		Vec3 extension = mMaxPosition - mMinPosition;

		u32 mostExtendingAxis = 0;
		if (extension.y() > extension.x())
		{
			mostExtendingAxis = 1;
		}

		if (extension.z() > extension.y() && extension.z() > extension.x())
		{
			mostExtendingAxis = 2;
		}

		return mostExtendingAxis;
	}

	static AABB generateAbsolutelyWrappedAABB()
	{
		constexpr f32 maxF32 = std::numeric_limits<f32>::max();
		constexpr f32 minF32 = std::numeric_limits<f32>::min();
		Vec3 minPos{maxF32, maxF32, maxF32};
		Vec3 maxPos{minF32, minF32, minF32};
		return AABB(minPos, maxPos);
	}

	static AABB generateWrapingAABB(const AABB& aabb0, const AABB& aabb1)
	{
		Vec3 minPosition
		{
			std::min(aabb0.getMinPosition()[0], aabb1.getMinPosition()[0]),
			std::min(aabb0.getMinPosition()[1], aabb1.getMinPosition()[1]),
			std::min(aabb0.getMinPosition()[2], aabb1.getMinPosition()[2])
		};

		Vec3 maxPosition
		{
			std::max(aabb0.getMaxPosition()[0], aabb1.getMaxPosition()[0]),
			std::max(aabb0.getMaxPosition()[1], aabb1.getMaxPosition()[1]),
			std::max(aabb0.getMaxPosition()[2], aabb1.getMaxPosition()[2])
		};

		return AABB(minPosition, maxPosition);
	}

	static AABB generateWrapingAABB(const AABB& aabb0, const Vec3& position)
	{
		Vec3 minPosition
		{
			std::min(aabb0.getMinPosition()[0], position[0]),
			std::min(aabb0.getMinPosition()[1], position[1]),
			std::min(aabb0.getMinPosition()[2], position[2])
		};

		Vec3 maxPosition
		{
			std::max(aabb0.getMaxPosition()[0], position[0]),
			std::max(aabb0.getMaxPosition()[1], position[1]),
			std::max(aabb0.getMaxPosition()[2], position[2])
		};

		return AABB(minPosition, maxPosition);
	}

private:
	Vec3 mMinPosition;
	Vec3 mMaxPosition;
};