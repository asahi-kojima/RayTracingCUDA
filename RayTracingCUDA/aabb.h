#pragma once
#include "common.h"
#include "matrix.h"
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

	static AABB transformAABB(const AABB& aabb, const Mat4& mat)
	{
		Vec3 minPosition(std::numeric_limits<f32>::max(), std::numeric_limits<f32>::max(), std::numeric_limits<f32>::max());
		Vec3 maxPosition(std::numeric_limits<f32>::min(), std::numeric_limits<f32>::min(), std::numeric_limits<f32>::min());
		for (s32 xi = 0; xi < 2; xi++)
		{
			for (s32 yi = 0; yi < 2; yi++)
			{
				for (s32 zi = 0; zi < 2; zi++)
				{
					Vec4 vertex{
						aabb.getMaxPosition()[0] * (xi) + aabb.getMinPosition()[0] * (1 - xi),
						aabb.getMaxPosition()[1] * (yi) + aabb.getMinPosition()[1] * (1 - yi),
						aabb.getMaxPosition()[2] * (zi) + aabb.getMinPosition()[2] * (1 - zi), 1};

					Vec4 transformedVertexAsVec4 = mat * vertex;
					Vec3 transformedVertex = transformedVertexAsVec4.extractXYZ();

					minPosition = Vec3
					{
						std::min(minPosition[0], transformedVertex[0]),
						std::min(minPosition[1], transformedVertex[1]),
						std::min(minPosition[2], transformedVertex[2])
					};

					maxPosition = Vec3
					{
						std::max(maxPosition[0], transformedVertex[0]),
						std::max(maxPosition[1], transformedVertex[1]),
						std::max(maxPosition[2], transformedVertex[2])
					};
				}
			}
		}

		return AABB(minPosition, maxPosition);
	}

private:
	Vec3 mMinPosition;
	Vec3 mMaxPosition;
};