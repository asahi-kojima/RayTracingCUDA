#pragma once
#include "common.h"
#include "vertex.h"
#include "aabb.h"

class Mesh
{
public:
	Mesh(const std::vector<Vertex>& vertices, const std::vector<u32> indices);

	Mesh(const Mesh& mesh);

	const std::vector<Vertex>& getVertexArray() const { return mVertexArray; }
	const std::vector<u32>& getIndexArray() const { return mIndexArray; }
	const std::vector<uint3>& getIndexArrayAsUint3() const { return mIndexArrayAsUint3; };
	AABB getAABB() const;

private:
	std::vector<Vertex> mVertexArray;
	std::vector<u32> mIndexArray;
	std::vector<uint3> mIndexArrayAsUint3;
	AABB mAABB;
};