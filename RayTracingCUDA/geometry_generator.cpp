#include "geometry_generator.h"

namespace
{
	Vertex getMidPoint(const Vertex& v0, const Vertex& v1)
	{
		const Vec3& p0 = v0.position;
		const Vec3& p1 = v1.position;

		const Vec3& n0 = v0.normal;
		const Vec3& n1 = v1.normal;


		Vec3 pos = (0.5f * (p0 + p1)).normalize() * ((p0.length() + p1.length()) / 2.0f);
		Vec3 normal = 0.5f * (n0 + n1);

		return Vertex(pos, normal);
	}

	void subDivide(std::vector<Vertex>& vertexArray, std::vector<u32>& indexArray)
	{
		std::vector<Vertex> vertexArrayCopy = vertexArray;
		std::vector<u32> indexArrayCopy = indexArray;

		vertexArray.resize(0);
		indexArray.resize(0);

		u32 numTriangles = static_cast<u32>(indexArrayCopy.size() / 3);

		for (u32 i = 0; i < numTriangles; i++)
		{
			Vertex v0 = vertexArrayCopy[indexArrayCopy[i * 3 + 0]];
			Vertex v1 = vertexArrayCopy[indexArrayCopy[i * 3 + 1]];
			Vertex v2 = vertexArrayCopy[indexArrayCopy[i * 3 + 2]];


			//中間点を生成
			Vertex m0 = getMidPoint(v0, v1);
			Vertex m1 = getMidPoint(v1, v2);
			Vertex m2 = getMidPoint(v0, v2);

			vertexArray.push_back(v0);
			vertexArray.push_back(v1);
			vertexArray.push_back(v2);
			vertexArray.push_back(m0);
			vertexArray.push_back(m1);
			vertexArray.push_back(m2);

			//頂点は4倍に増える。
			//インデックスは６個に増えたので、6をオフセットで掛けている。
			indexArray.push_back(i * 6 + 0);
			indexArray.push_back(i * 6 + 3);
			indexArray.push_back(i * 6 + 5);

			indexArray.push_back(i * 6 + 3);
			indexArray.push_back(i * 6 + 4);
			indexArray.push_back(i * 6 + 5);

			indexArray.push_back(i * 6 + 5);
			indexArray.push_back(i * 6 + 4);
			indexArray.push_back(i * 6 + 2);

			indexArray.push_back(i * 6 + 3);
			indexArray.push_back(i * 6 + 1);
			indexArray.push_back(i * 6 + 4);
		}
	}
}

Mesh GeometryGenerator::tetrahedronGenerator()
{
	std::vector<Vec3> positions
	{
		Vec3(0.0f,      +1.0f, 0.0f),
		Vec3(0,         -0.5f, +std::sqrtf(3) / 2.0f),
		Vec3(+3.0f / 4, -0.5f, -std::sqrtf(3) / 4.0f),
		Vec3(-3.0f / 4, -0.5f, -std::sqrtf(3) / 4.0f)
	};



	std::vector<Vertex> vertexArray;

	for (u32 i = 0; i < 4; i++) 
	{
		vertexArray.emplace_back(positions[i], positions[i].normalize());
	}
	
	std::vector<u32> indexArray
	{
		0, 1, 2, 
		0, 3, 1,
		0, 2, 3,
		1, 3, 2
	};

	return Mesh(vertexArray, indexArray);
}

Mesh GeometryGenerator::octahedronGenerator()
{
	std::vector<Vec3> positions
	{
		Vec3(0.0f, +1.0f,  0.0f),
		Vec3(0.0f,  0.0f, +1.0f),
		Vec3(+1.0f, 0.0f,  0.0f),
		Vec3(0.0f,  0.0f, -1.0f),
		Vec3(-1.0f, 0.0f,  0.0f),
		Vec3(0.0f, -1.0f,  0.0f)
	};


	std::vector<Vertex> vertexArray;

	for (u32 i = 0; i < 6; ++i)
	{
		vertexArray.emplace_back(positions[i], positions[i].normalize());
	}

	std::vector<u32> indexArray
	{
		0, 1, 2,
		0, 2, 3,
		0, 3, 4,
		0, 4, 1,
		5, 2, 1,
		5, 3, 2,
		5, 4, 3,
		5, 1, 4
	};

	return Mesh(vertexArray, indexArray);
}

Mesh GeometryGenerator::boxGenerator()
{
	const f32 halfEdgeLength = 1.0f;

	std::vector<Vertex> vertexArray{ 24 };

	vertexArray[0] = Vertex(-halfEdgeLength, -halfEdgeLength, -halfEdgeLength, 0.0f, 0.0f, -1.0f);
	vertexArray[1] = Vertex(-halfEdgeLength, +halfEdgeLength, -halfEdgeLength, 0.0f, 0.0f, -1.0f);
	vertexArray[2] = Vertex(+halfEdgeLength, +halfEdgeLength, -halfEdgeLength, 0.0f, 0.0f, -1.0f);
	vertexArray[3] = Vertex(+halfEdgeLength, -halfEdgeLength, -halfEdgeLength, 0.0f, 0.0f, -1.0f);

	vertexArray[4] = Vertex(-halfEdgeLength, -halfEdgeLength, +halfEdgeLength, 0.0f, 0.0f, +1.0f);
	vertexArray[5] = Vertex(+halfEdgeLength, -halfEdgeLength, +halfEdgeLength, 0.0f, 0.0f, +1.0f);
	vertexArray[6] = Vertex(+halfEdgeLength, +halfEdgeLength, +halfEdgeLength, 0.0f, 0.0f, +1.0f);
	vertexArray[7] = Vertex(-halfEdgeLength, +halfEdgeLength, +halfEdgeLength, 0.0f, 0.0f, +1.0f);

	vertexArray[8] = Vertex(-halfEdgeLength, +halfEdgeLength, -halfEdgeLength, 0.0f, +1.0f, 0.0f);
	vertexArray[9] = Vertex(-halfEdgeLength, +halfEdgeLength, +halfEdgeLength, 0.0f, +1.0f, 0.0f);
	vertexArray[10] = Vertex(+halfEdgeLength, +halfEdgeLength, +halfEdgeLength, 0.0f, +1.0f, 0.0f);
	vertexArray[11] = Vertex(+halfEdgeLength, +halfEdgeLength, -halfEdgeLength, 0.0f, +1.0f, 0.0f);

	vertexArray[12] = Vertex(-halfEdgeLength, -halfEdgeLength, -halfEdgeLength, 0.0f, -1.0f, 0.0f);
	vertexArray[13] = Vertex(+halfEdgeLength, -halfEdgeLength, -halfEdgeLength, 0.0f, -1.0f, 0.0f);
	vertexArray[14] = Vertex(+halfEdgeLength, -halfEdgeLength, +halfEdgeLength, 0.0f, -1.0f, 0.0f);
	vertexArray[15] = Vertex(-halfEdgeLength, -halfEdgeLength, +halfEdgeLength, 0.0f, -1.0f, 0.0f);

	vertexArray[16] = Vertex(-halfEdgeLength, -halfEdgeLength, +halfEdgeLength, -1.0f, 0.0f, 0.0f);
	vertexArray[17] = Vertex(-halfEdgeLength, +halfEdgeLength, +halfEdgeLength, -1.0f, 0.0f, 0.0f);
	vertexArray[18] = Vertex(-halfEdgeLength, +halfEdgeLength, -halfEdgeLength, -1.0f, 0.0f, 0.0f);
	vertexArray[19] = Vertex(-halfEdgeLength, -halfEdgeLength, -halfEdgeLength, -1.0f, 0.0f, 0.0f);

	vertexArray[20] = Vertex(+halfEdgeLength, -halfEdgeLength, -halfEdgeLength, +1.0f, 0.0f, 0.0f);
	vertexArray[21] = Vertex(+halfEdgeLength, +halfEdgeLength, -halfEdgeLength, +1.0f, 0.0f, 0.0f);
	vertexArray[22] = Vertex(+halfEdgeLength, +halfEdgeLength, +halfEdgeLength, +1.0f, 0.0f, 0.0f);
	vertexArray[23] = Vertex(+halfEdgeLength, -halfEdgeLength, +halfEdgeLength, +1.0f, 0.0f, 0.0f);


	std::vector<u32> indexArray(36);
	indexArray[0] = 0;	indexArray[1] = 1;	indexArray[2] = 2;
	indexArray[3] = 0;	indexArray[4] = 2;	indexArray[5] = 3;

	indexArray[6] = 4;	indexArray[7] = 5;	indexArray[8] = 6;
	indexArray[9] = 4;	indexArray[10] = 6;	indexArray[11] = 7;

	indexArray[12] = 8;	indexArray[13] = 9;	indexArray[14] = 10;
	indexArray[15] = 8;	indexArray[16] = 10;	indexArray[17] = 11;

	indexArray[18] = 12;	indexArray[19] = 13;	indexArray[20] = 14;
	indexArray[21] = 12;	indexArray[22] = 14;	indexArray[23] = 15;

	indexArray[24] = 16;	indexArray[25] = 17;	indexArray[26] = 18;
	indexArray[27] = 16;	indexArray[28] = 18;	indexArray[29] = 19;

	indexArray[30] = 20;	indexArray[31] = 21;	indexArray[32] = 22;
	indexArray[33] = 20;	indexArray[34] = 22;	indexArray[35] = 23;


	return Mesh(vertexArray, indexArray);
}

Mesh GeometryGenerator::sphereGenerator(const u32 stackCount, const u32 sliceCount)
{
	const f32 radius = 1.0f;

	Vertex topVertex(0.0f, +radius, 0.0f, 0.0f, +1.0f, 0.0f);
	Vertex bottomVertex(0.0f, -radius, 0.0f, 0.0f, -1.0f, 0.0f);

	std::vector<Vertex> vertexArray;
	vertexArray.push_back(topVertex);

	const f32 dTheta = M_PI / (stackCount + 1);
	const f32 dPhi   = 2.0f * M_PI / sliceCount;

	// 中間のスタック（極以外）
	for (u32 i = 0; i < stackCount; i++)
	{
		f32 theta = (i + 1) * dTheta;
		for (u32 j = 0; j < sliceCount; j++)
		{
			f32 phi = j * dPhi;
			Vertex v;
			v.position[0] = radius * sinf(theta) * cosf(phi);
			v.position[1] = radius * cosf(theta);
			v.position[2] = radius * sinf(theta) * sinf(phi);
			v.normal[0] = sinf(theta) * cosf(phi);
			v.normal[1] = cosf(theta);
			v.normal[2] = sinf(theta) * sinf(phi);
			vertexArray.push_back(v);
		}
	}
	vertexArray.push_back(bottomVertex);

	std::vector<u32> indexArray;


	for (u32 i = 0; i < sliceCount; i++)
	{
		indexArray.push_back(0);
		indexArray.push_back(1 + i);
		indexArray.push_back(1 + (i + 1) % sliceCount);
	}

	for (u32 i = 0; i < stackCount - 1; i++)
	{
		for (u32 j = 0; j < sliceCount; j++)
		{
			u32 current = 1 + i * sliceCount + j;
			u32 next = 1 + i * sliceCount + (j + 1) % sliceCount;
			u32 below = 1 + (i + 1) * sliceCount + j;
			u32 belowNext = 1 + (i + 1) * sliceCount + (j + 1) % sliceCount;

			indexArray.push_back(current);
			indexArray.push_back(below);
			indexArray.push_back(next);

			indexArray.push_back(next);
			indexArray.push_back(below);
			indexArray.push_back(belowNext);
		}
	}


	u32 southPoleIndex = static_cast<u32>(vertexArray.size() - 1);
	u32 baseIndex = southPoleIndex - sliceCount;
	for (u32 i = 0; i < sliceCount; i++)
	{
		indexArray.push_back(southPoleIndex);
		indexArray.push_back(baseIndex + (i + 1) % sliceCount);
		indexArray.push_back(baseIndex + i);
	}

	return Mesh(vertexArray, indexArray);
}

Mesh GeometryGenerator::geoSphereGenerator(const u32 subDivisionScale)
{
	const f32 X = 0.525731f;
	const f32 Z = 0.850651f;

	std::vector<Vertex> vertexArray
	{
		Vertex(-X, 0.0f, +Z, 0, 0, 0), Vertex(+X, 0.0f, +Z, 0, 0, 0),
		Vertex(-X, 0.0f, -Z, 0, 0, 0), Vertex(+X, 0.0f, -Z, 0, 0, 0),
		Vertex(0.0f, +Z, +X, 0, 0, 0), Vertex(0.0f, +Z, -X, 0, 0, 0),
		Vertex(0.0f, -Z, +X, 0, 0, 0), Vertex(0.0f, -Z, -X, 0, 0, 0),
		Vertex(+Z, +X, 0.0f, 0, 0, 0), Vertex(-Z, +X, 0.0f, 0, 0, 0),
		Vertex(+Z, -X, 0.0f, 0, 0, 0), Vertex(-Z, -X, 0.0f, 0, 0, 0)
	};

	std::vector<u32> indexArray
	{
		1,4,0,	4,9,0,	4,5,9,	8,5,4,	1,8,4,
		1,10,8,	10,3,8,	8,3,5,	3,2,5,	3,7,2,
		3,10,7,	10,6,7,	6,11,7,	6,0,11,	6,1,0,
		10,1,6,	11,0,9,	2,11,9,	5,2,9,	11,2,7
	};



	for (s32 i = 0; i < std::min<u32>(subDivisionScale, 5u); i++)
	{
		subDivide(vertexArray, indexArray);
	}

	for (Vertex& vertex : vertexArray)
	{
		Vec3 normal = vertex.position.normalize();
		vertex.normal = normal;
	}

	return Mesh(vertexArray, indexArray);
}