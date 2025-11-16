#include "mesh.h"


namespace
{
	AABB generateAABB(const std::vector<Vertex>& vertices)
	{
		Vec3 minPos = Vec3::generateMaximumLengthVector();
		Vec3 maxPos = Vec3::generateMinimumLengthVector();

		for (const auto& vertex : vertices)
		{
			minPos.x() = fminf(minPos.x(), vertex.position.x());
			minPos.y() = fminf(minPos.y(), vertex.position.y());
			minPos.z() = fminf(minPos.z(), vertex.position.z());

			maxPos.x() = fmaxf(maxPos.x(), vertex.position.x());
			maxPos.y() = fmaxf(maxPos.y(), vertex.position.y());
			maxPos.z() = fmaxf(maxPos.z(), vertex.position.z());
		}

		const Vec3 padding{ 1e-5f, 1e-5f, 1e-5f };
		return AABB(minPos - padding, maxPos + padding);
	}

}

Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<u32> indices)
	: mVertexArray(vertices)
	, mIndexArray(indices)
	, mIndexArrayAsUint3(0)
	, mAABB(generateAABB(vertices))
{
	assert(mIndexArray.size() % 3 == 0);

	for (u32 i = 0; i < mIndexArray.size() / 3; i++)
	{
		uint3 index{
			mIndexArray[3 * i + 0],
			mIndexArray[3 * i + 1],
			mIndexArray[3 * i + 2]
		};

		mIndexArrayAsUint3.push_back(index);
	}
}

Mesh::Mesh(const Mesh& mesh)
	: mVertexArray(mesh.mVertexArray)
	, mIndexArray(mesh.mIndexArray)
	, mIndexArrayAsUint3(mesh.mIndexArrayAsUint3)
	, mAABB(mesh.mAABB)
{
}

AABB Mesh::getAABB() const
{
	Vec3 minPosition = mVertexArray[0].position;
	Vec3 maxPosition = mVertexArray[0].position;
	for (const Vertex& vertex : mVertexArray)
	{

		minPosition[0] = fminf(minPosition[0], vertex.position[0]);
		minPosition[1] = fminf(minPosition[1], vertex.position[1]);
		minPosition[2] = fminf(minPosition[2], vertex.position[2]);
		maxPosition[0] = fmaxf(maxPosition[0], vertex.position[0]);
		maxPosition[1] = fmaxf(maxPosition[1], vertex.position[1]);
		maxPosition[2] = fmaxf(maxPosition[2], vertex.position[2]);
	}

	return AABB(minPosition, maxPosition);
}