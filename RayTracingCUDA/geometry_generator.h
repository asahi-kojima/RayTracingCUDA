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
	static Mesh planeGenerator(const u32 divisionCount = 1);
	static Mesh coneGenerator(const u32 divisionCount = 20);
	static Mesh cylinderGenerator(const u32 divisionCount = 20);
	static Mesh torusGenerator(const f32 minorRadiusRaio = 0.3f, const u32 majorDivisionCount = 20, const u32 minorDivisionCount = 12);
};
