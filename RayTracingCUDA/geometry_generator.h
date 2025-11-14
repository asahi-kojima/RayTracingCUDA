#pragma once
#include "mesh.h"
#include "vertex.h"


class GeometryGenerator
{
public:
	static Mesh tetrahedronGenerator();
	static Mesh octahedronGenerator();
	static Mesh boxGenerator();
	static Mesh sphereGenerator(const u32 stackCount = 5, const u32 sliceCount = 5);
	static Mesh geoSphereGenerator(const u32 subDivisionScale = 3);
};
