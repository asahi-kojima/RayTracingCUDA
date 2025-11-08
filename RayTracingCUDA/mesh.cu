#include "mesh.h"


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