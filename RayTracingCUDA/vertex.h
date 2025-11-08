#pragma once
#include "vector.h"

class Vertex
{
public:
	Vertex() = default;
	Vertex(const Vec3& p,const Vec3& n)
		: position(p)
		, normal(n)
	{}

	Vertex(
		f32 px, f32 py, f32 pz,
		f32 nx, f32 ny, f32 nz)
		: position(px, py, pz)
		, normal(nx, ny, nz)
	{}

	Vertex(const Vertex& v)
		: position(v.position)
		, normal(v.normal)
	{
	}

	Vec3 position;
	Vec3 normal;
};