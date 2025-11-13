#pragma once
#include <unordered_map>
#include "mesh.h"
#include "material.h"
#include "object_and_group.h"
#include "camera.h"

struct DeviceInstanceData
{
	Mat4 transformMat;
	Mat4 invTransformMat;
	Mat4 normalTransformMat;
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

struct PrimitiveInfo
{
	u32 primitiveID;
	AABB aabb;
	Vec3 centroid;
};

struct BlasInfo//delete
{
	u32 vertexOffset;
	u32 triangleIndexOffset;
	u32 blasRootNodeIndex;
};


struct RayTracingDataOnCPU
{
	// Meshに対応
	std::vector<float3> vertexArray;
	std::vector<uint3>  triangleIndexArray;
	std::vector<float3> normalArray;

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



struct GpuRayTracingLaunchParams
{
	//--------------------------------------
	// シーンを構成するオブジェクト関係
	//--------------------------------------
	// Meshに対応
	float3* vertexArray;
	uint3* triangleIndexArray; // Triangleを指定するためのID＝ポリゴンのID
	float3* normalArray; // ↑のポリゴンの法線

	// マテリアルに対応
	Material* materialArray;

	// オブジェクトに対応：どのメッシュ/マテリアルを使うかなど。
	DeviceInstanceData* instanceDataArray;

	// TLAS/BLAS
	BVHNode* blasArray;
	BVHNode* tlasArray;


	u32 vertexCount = 0;
	u32 indexCount = 0;
	u32 normalCount = 0;
	u32 materialCount = 0;
	u32 instanceCount = 0;
	u32 blasCount = 0;
	u32 tlasCount = 0;

	u32 pixelSizeVertical;
	u32 pixelSizeHorizontal;

	f32 invPixelSizeVertical;
	f32 invPixelSizeHorizontal;

	Color* renderTargetImageArray;

	u32 frameCount;

	Camera camera;
};

/// <summary>
/// 現状では固定のメモリ領域上に構築することを前提としているシーンデータ
/// 動的に必要なメモリが変わる状況には後に対応予定である。
/// </summary>
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
	Result initLaunchParams();
	Result render();
	Result update();

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


	GpuRayTracingLaunchParams mGpuRayTracingLaunchParamsHostSide;

	//GPUレイトレーシングで必要になるデータを作るためのサブデータやヘルパー関数達


	void buildVertexIndexBlas();
	u32 buildBlasBVHNode(const Mesh& mesh, std::vector<uint3>& sortedIndexArray);
	u32 buildBlasBVHNodeRecursively(std::vector<BVHNode>& nodeArray, const Mesh& mesh, std::vector<PrimitiveInfo>& primitiveInfoArray, const u32 start, const u32 end);

	void buildInstanceData();
	void recursiveBuildInstanceData(std::vector<DeviceInstanceData>& instanceDataArray, const Group& group, const Mat4& currentTransformMat, const Mat4& currentInvTransformMat, const Mat4& currentInvTransposedTransformMat);


	void buildTlasBVHNode();
	u32 buildTlasBVHNodeRecursively(const u32 start, const u32 end);
};