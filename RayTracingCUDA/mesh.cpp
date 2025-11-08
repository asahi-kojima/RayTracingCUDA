#include "mesh.h"

AABB generateAABB(const std::vector<Vertex>& vertices)
{
	Vec3 minPos = vertices[0].position;
	Vec3 maxPos = vertices[0].position;

	for (const auto& vertex : vertices)
	{
		minPos.x() = fminf(minPos.x(), vertex.position.x());
		minPos.y() = fminf(minPos.y(), vertex.position.y());
		minPos.z() = fminf(minPos.z(), vertex.position.z());

		maxPos.x() = fmaxf(maxPos.x(), vertex.position.x());
		maxPos.y() = fmaxf(maxPos.y(), vertex.position.y());
		maxPos.z() = fmaxf(maxPos.z(), vertex.position.z());
	}

	return AABB(minPos, maxPos);
}

Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<u32> indices)
	: mVertexArray(vertices)
	, mIndexArray(indices)
	, mAABB(generateAABB(vertices))
{
}

Mesh::Mesh(const Mesh& mesh)
	: mVertexArray(mesh.mVertexArray)
	, mIndexArray(mesh.mIndexArray)
	, mAABB(mesh.mAABB)
{
}