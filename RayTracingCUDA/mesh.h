#pragma once
#include "common.h"
#include "vertex.h"
#include "aabb.h"

class Mesh
{
public:
	Mesh() = default;
	Mesh(const std::vector<Vertex>& vertices, const std::vector<u32> indices)
		: mVertexArray(vertices)
		, mIndexArray(indices)
	{
	}

	Mesh(const Mesh& mesh)
		: mVertexArray(mesh.mVertexArray)
		, mIndexArray(mesh.mIndexArray)
		, mAABB(mesh.mAABB)
	{
	}


	AABB getAABB() const;

private:
	std::vector<Vertex> mVertexArray;
	std::vector<u32> mIndexArray;
	AABB mAABB;
};