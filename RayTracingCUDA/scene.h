#pragma once
#include <unordered_map>
#include "mesh.h"
#include "material.h"
#include "object_and_group.h"

struct DeviceInstanceData
{
	Mat4 transformMat;
	Mat4 invTransformMat;
	AABB aabb;

	u32 blasRootNodeIndex;

	u32 vertexOffset;
	u32 indexOffset;

	u32 materialID;
};


struct BVHNode
{
	AABB aabb;
	u32 leftChildOffset;
	u32 rightChildOffset;

	u32 firstPrimitiveOffset;
	u32 primitiveCount;
};

// RayTracingをGPUで行うにあたり必要なデータ
// これを準備すれば十分である。
struct RayTracingDataOnGPU
{
	// Meshに対応
	float3* vertexArray;
	uint3* indexArray;

	// マテリアルに対応
	Material* materialArray;

	// オブジェクトに対応：どのメッシュ/マテリアルを使うかなど。
	DeviceInstanceData* instanceDataArray;

	// TLAS/BLAS
	BVHNode* blasArray;
	BVHNode* tlasArray;
};

struct BlasInfo//delete
{
	u32 vertexOffset;
	u32 indexOffset;
	u32 blasRootNodeIndex;
};

// RayTracingDataOnGPUへのコピーを念頭に置いているので、基本同じ構造を持つ
struct RayTracingDataOnCPU
{
	// Meshに対応
	std::vector<float3> vertexArray;
	std::vector<uint3> indexArray;

	// マテリアルに対応
	std::vector<Material> materialArray;

	// オブジェクトに対応：どのメッシュ/マテリアルを使うかなど。
	std::vector<DeviceInstanceData> instanceDataArray;

	// TLAS/BLAS
	std::vector<BVHNode> blasArray;
	std::vector<BVHNode> tlasArray;


	//補助変数
	std::vector<BlasInfo> blasInfoArray;
};






class Scene
{
public:
	Scene()
		: mMeshArray()
		, mMaterialArray()
		, mRootGroup("root")
	{ }

	s32 addMesh(const std::string& meshName ,const Mesh& mesh);
	s32 addMaterial(const std::string& materialName, const Material& material);

	Result addObject(const Object& object, const Transform& transform = Transform::identity(), const std::string& newName = std::string(""));
	Result addGroup(const Group& group, const Transform& transform = Transform::identity(), const std::string& newName = std::string(""));

	Result build();
	Result render();

private:
	// GPU上でのレイトレーシングとは独立して存在するデータ
	std::vector<Mesh> mMeshArray;
	std::vector<Material> mMaterialArray;

	std::unordered_map<std::string, u32> mMeshNameToIdMap;
	std::unordered_map<std::string, u32> mMaterialNameToIdMap;

	Group mRootGroup;

	//ここからRayTracingに関係したセクション
	// GPU上でレイトレーシングをするために必要なデータ
	RayTracingDataOnCPU mRayTracingDataOnCPU;
	RayTracingDataOnGPU mRayTracingDataOnGPU;


	//GPUレイトレーシングで必要になるデータを作るためのサブデータやヘルパー関数達


	void buildVertexIndexBlas();
	u32 buildBVHNode(const Mesh& mesh, std::vector<uint3>& sortedIndexArray);
	void recursiveBuildInstanceData(std::vector<DeviceInstanceData>& instanceDataArray, const Group& group, const Mat4& currentTransformMat, const Mat4& currentInvTransformMat);
	void buildInstanceData();
};