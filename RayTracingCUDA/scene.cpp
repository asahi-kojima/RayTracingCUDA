#include <format>
#include <algorithm>
#include <iostream>
#include "util.h"
#include "scene.h"

namespace
{

	AABB generateAABBFromTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2)
	{
		Vec3 minPosition{
			std::min({v0[0], v1[0], v2[0]}),
			std::min({v0[1], v1[1], v2[1]}),
			std::min({v0[2], v1[2], v2[2]})
		};
		
		Vec3 maxPosition{
			std::max({v0[0], v1[0], v2[0]}),
			std::max({v0[1], v1[1], v2[1]}),
			std::max({v0[2], v1[2], v2[2]})
		};

		return AABB{minPosition, maxPosition};
	}
}

s32 Scene::addMesh(const std::string& meshName, const Mesh& mesh)
{
	if (mMeshNameToIdMap.count(meshName) > 0)
	{
		return -1;
	}

	u32 newId = mMeshArray.size();
	
	mMeshArray.push_back(mesh);

	mMeshNameToIdMap[meshName] = newId;

	return newId;
}

s32 Scene::addMaterial(const std::string& materialName, const Material& material)
{
	if (mMaterialNameToIdMap.count(materialName) > 0)
	{
		return -1;
	}

	u32 newId = mMaterialArray.size();

	mMaterialArray.push_back(material);

	mMaterialNameToIdMap[materialName] = newId;

	return newId;
}

Result Scene::addObject(const Object& object, const Transform& transform, const std::string& newName)
{
	//objectの追加時に、MeshとMaterialが存在している正規の物か確認する
	const std::string& meshName = object.getMeshName();
	if (mMeshNameToIdMap.count(meshName) == 0)
	{
		return Result(false, std::format("Specified mesh [{}] does not exist in the scene.", meshName.c_str()));
	}

	const std::string& materialName = object.getMaterialName();
	if (mMaterialNameToIdMap.count(materialName) == 0)
	{
		return Result(false, std::format("Specified Material [{}] does not exist in the scene.", materialName.c_str()));

	}

	//確認が取れたら登録する
	Result result = mRootGroup.addChildObject(object, transform, newName);

	return result;
}

Result Scene::addGroup(const Group& group, const Transform& transform, const std::string& newName)
{
	//メッシュが正しいか確認
	std::vector<std::string> meshNameArray;
	group.getAllMeshNamesChildrenHave(meshNameArray);
	for (const std::string& meshName : meshNameArray)
	{
		if (mMeshNameToIdMap.count(meshName) == 0)
		{
			return Result(false, std::format("Specified mesh [%s] does not exist in the scene.", meshName.c_str()));
		}
	}

	//マテリアルが正しいか確認
	std::vector<std::string> materialNameArray;
	group.getAllMateialNamesChildrenHave(materialNameArray);
	for (const std::string& materialName : materialNameArray)
	{
		if (mMaterialNameToIdMap.count(materialName) == 0)
		{
			return Result(false, std::format("Specified Material [%s] does not exist in the scene.", materialName.c_str()));
		}
	}

	//確認が取れたら登録する
	Result result = mRootGroup.addChildGroup(group, transform, newName);

	return result;
}



Result Scene::build()
{
	std::cout << "===================================================" << std::endl;
	std::cout << "               Scene Build Start                   " << std::endl;
	std::cout << "===================================================" << std::endl;
	std::cout << std::format("There exists {} obects in this Scene", mRootGroup.getDescendantObjectCount()) << std::endl;
	std::cout << std::format("There exists {} meshed in this Scene", mMeshArray.size()) << std::endl;
	std::cout << std::format("There exists {} materials in this Scene", mMaterialArray.size()) << std::endl;


	std::cout << "===================================================" << std::endl;
	std::cout << "       Prepare Essetial Data For RayTracing        " << std::endl;
	std::cout << "===================================================" << std::endl;
	/*
		以下を埋めていくことを念頭に進める。
		// Meshに対応
		(1)float3* vertexArray;
		(1)uint3* indexArray;

		// マテリアルに対応
		(2)Material* materialArray;

		// オブジェクトに対応：どのメッシュ/マテリアルを使うかなど。
		(3)DeviceInstanceData* instanceDataArray;

		// TLAS/BLAS
		(1)BVHNode* blasArray;
		(4)BVHNode* tlasArray;
	*/

	//まずはトランスフォームのアップデートを行う// todo : 一般のupdateに昇格させたほうがいいかも
	mRootGroup.updateAllChildrenTransform();

	//オブジェクトの平坦化を行う。
	//std::vector<>
	//mRootGroup.flatten();

	// ----------------------------------------------------
	// (0) CPU側の情報の消去
	// ----------------------------------------------------
	mRayTracingDataOnCPU.vertexArray.resize(0);
	mRayTracingDataOnCPU.indexArray.resize(0);
	mRayTracingDataOnCPU.normalArray.resize(0);
	mRayTracingDataOnCPU.materialArray.resize(0);
	mRayTracingDataOnCPU.instanceDataArray.resize(0);
	mRayTracingDataOnCPU.blasArray.resize(0);
	mRayTracingDataOnCPU.tlasArray.resize(0);
	
	mRayTracingDataOnCPU.blasInfoArray.resize(0);


	// ----------------------------------------------------
	// (1) Meshデータから頂点、インデックス、BLASの構築
	// ----------------------------------------------------
	buildVertexIndexBlas();


	// ----------------------------------------------------
	// (2) Materialのコピー
	// ----------------------------------------------------
	mRayTracingDataOnCPU.materialArray = mMaterialArray;


	// ----------------------------------------------------
	// (3) InstanceDataの構築
	// ----------------------------------------------------
	buildInstanceData();


	// ----------------------------------------------------
	// (4) TLASの構築
	// ----------------------------------------------------
	buildTlasBVHNode();



	return Result();
}







//=========================================================================================
// BLAS構築に関わる関数達
//=========================================================================================
void Scene::buildVertexIndexBlas()
{
	//--------------------------------------------------------------------
	// この作業で全メッシュの頂点データ、インデックスデータ、BLASデータが
	// 一列に並ぶことになる。
	//--------------------------------------------------------------------
	for (const auto& mesh : mMeshArray)
	{
		const std::vector<Vertex>& vertexArray = mesh.getVertexArray();
		std::vector<uint3> sortdIndexArray;

		// MeshからBLASを構築する
		const u32 blasRootNode = buildBlasBVHNode(mesh, sortdIndexArray);

		BlasInfo blasInfo;
		{
			blasInfo.vertexOffset = mRayTracingDataOnCPU.vertexArray.size();
			blasInfo.indexOffset = mRayTracingDataOnCPU.indexArray.size();
			blasInfo.blasRootNodeIndex = blasRootNode;
		}

		// このメッシュの頂点データの格納
		for (const auto& vertex : vertexArray)
		{
			float3 vertexPosition = vertex.position.toFloat3();
			mRayTracingDataOnCPU.vertexArray.push_back(vertexPosition);
		}

		// このメッシュの三角形インデックスを格納
		for (u32 triangleID = 0; triangleID < sortdIndexArray.size(); triangleID++)
		{
			uint3 index = sortdIndexArray[triangleID];
			mRayTracingDataOnCPU.indexArray.push_back(index);

			// 法線データもここで格納しておく
			const Vec3& v0 = vertexArray[index.x].position;
			const Vec3& v1 = vertexArray[index.y].position;
			const Vec3& v2 = vertexArray[index.z].position;
			Vec3 normal = Vec3::normalize(Vec3::cross(v1 - v0, v2 - v0));
			mRayTracingDataOnCPU.normalArray.push_back(normal.toFloat3());
		}

		mRayTracingDataOnCPU.blasInfoArray.push_back(blasInfo);
	}
}


namespace
{
	struct PrimitiveInfo
	{
		u32 primitiveID;
		AABB aabb;
		Vec3 centroid;
	};

	u32 buildBlasBVHNodeRecursively(std::vector<BVHNode>& nodeArray, const Mesh& mesh, std::vector<PrimitiveInfo>& primitiveInfoArray, const u32 start, const u32 end)
	{
		const u32 nodeIndex = nodeArray.size();
		nodeArray.emplace_back();
		//BVHNode& currentNode = nodeArray[nodeIndex];

		const u32 primitiveCount = end - start;

		//------------------------------------------------------------------------------
		// このノードが管理するAABBを計算する : ([start, end)を包括するAABBを算出)
		//------------------------------------------------------------------------------
		AABB aabb = AABB::generateAbsolutelyWrappedAABB();
		for (u32 i = start; i < end; i++)
		{
			aabb = AABB::generateWrapingAABB(aabb, primitiveInfoArray[i].aabb);
		}
		nodeArray[nodeIndex].aabb = aabb;

		//------------------------------------------------------------------------------
		// 末端であればここでリターン
		//------------------------------------------------------------------------------
		const u32 maxPrimitiveCount = 4;
		if (primitiveCount <= maxPrimitiveCount)
		{
			nodeArray[nodeIndex].primitiveCount = primitiveCount;
			nodeArray[nodeIndex].firstPrimitiveOffset = start;
			return nodeIndex;
		}

		//------------------------------------------------------------------------------
		// 分割をしていく
		//------------------------------------------------------------------------------
		AABB centroidAABB = AABB::generateAbsolutelyWrappedAABB();
		for (u32 i = start; i < end; i++)
		{
			centroidAABB = AABB::generateWrapingAABB(centroidAABB, primitiveInfoArray[i].centroid);
		}

		//------------------------------------------------------------------------------
		// 一番空間的に広がっている軸を探す
		//------------------------------------------------------------------------------
		const u32 splitAxis = centroidAABB.getMostExtendingAxis();
		const f32 splitPoint = (centroidAABB.getMinPosition()[splitAxis] + centroidAABB.getMaxPosition()[splitAxis]) / 2.0f;

		//------------------------------------------------------------------------------
		// プリミティブ配列のソートを行う。
		//------------------------------------------------------------------------------
		auto midiumIter = std::partition(
			primitiveInfoArray.begin() + start, 
			primitiveInfoArray.begin() + end, 
			[splitAxis, splitPoint](const PrimitiveInfo& primitiveInfo)
			{
				return primitiveInfo.centroid[splitAxis] < splitPoint;
			}
		);

		u32 midium = std::distance(primitiveInfoArray.begin(), midiumIter);

		//分割が上手くできなかったとき
		if (midium == start || midium == end)
		{
			midium = start + primitiveCount / 2;
		}

		const u32 leftChildOffset = buildBlasBVHNodeRecursively(nodeArray, mesh, primitiveInfoArray, start, midium);
		const u32 rightChildOffset = buildBlasBVHNodeRecursively(nodeArray, mesh, primitiveInfoArray, midium, end);

		nodeArray[nodeIndex].primitiveCount = 0;
		nodeArray[nodeIndex].leftChildOffset = leftChildOffset;
		nodeArray[nodeIndex].rightChildOffset = rightChildOffset;

		return nodeIndex;
	}
}


u32 Scene::buildBlasBVHNode(const Mesh& mesh, std::vector<uint3>& sortedIndexArray)
{
	const std::vector<Vertex>& vertexArray = mesh.getVertexArray();
	const std::vector<uint3>& indexArray = mesh.getIndexArrayAsUint3();

	
	//ここでいうプリミティブとはメッシュを構成する三角形の事である。
	std::vector<PrimitiveInfo> primitiveInfoArray(indexArray.size());

	for (u32 i = 0; i < indexArray.size(); i++)
	{
		primitiveInfoArray[i].primitiveID = i;
		primitiveInfoArray[i].aabb = generateAABBFromTriangle(
			vertexArray[indexArray[i].x].position, 
			vertexArray[indexArray[i].y].position, 
			vertexArray[indexArray[i].z].position);
		primitiveInfoArray[i].centroid = (primitiveInfoArray[i].aabb.getMinPosition() + primitiveInfoArray[i].aabb.getMaxPosition()) / 2.0f;
	}


	const u32 rootBvhNodeIndex = buildBlasBVHNodeRecursively(mRayTracingDataOnCPU.blasArray, mesh, primitiveInfoArray, 0, primitiveInfoArray.size());

	sortedIndexArray.resize(primitiveInfoArray.size());
	for (u32 i = 0; i < primitiveInfoArray.size(); i++)
	{
		u32 originalPrimitiveID = primitiveInfoArray[i].primitiveID;
		sortedIndexArray[i] = indexArray[originalPrimitiveID];
	}

	return rootBvhNodeIndex;
}


//=========================================================================================
// インスタンスデータ構築のための関数
//=========================================================================================
void Scene::buildInstanceData()
{
	Mat4 transformMat = mRootGroup.getTransform().getTransformMatrix();
	Mat4 invTransformMat = mRootGroup.getTransform().getInvTransformMatrix();
	Mat4 invTransposedTransformMat = mRootGroup.getTransform().getInvTransposeTransformMatrix();
	
	recursiveBuildInstanceData(mRayTracingDataOnCPU.instanceDataArray, mRootGroup, transformMat, invTransformMat, invTransposedTransformMat);


}

void Scene::recursiveBuildInstanceData(std::vector<DeviceInstanceData>& instanceDataArray, const Group& group, const Mat4& currentTransformMat, const Mat4& currentInvTransformMat, const Mat4& currentInvTransposedTransformMat)
{
	for (const Object& childObject : group.getChildObjectArray())
	{
		Mat4 transformMat = childObject.getTransform().getTransformMatrix();
		Mat4 invTransformMat = childObject.getTransform().getInvTransformMatrix();
		Mat4 invTransposedTransformMat = childObject.getTransform().getInvTransposeTransformMatrix();
		
		const u32 meshID = mMeshNameToIdMap[childObject.getMeshName()];
		const u32 materialID = mMaterialNameToIdMap[childObject.getMaterialName()];

		AABB referenceMeshAABB = mMeshArray[meshID].getAABB();

		const BlasInfo& blasInfo = mRayTracingDataOnCPU.blasInfoArray[meshID];


		DeviceInstanceData instanceData
		{
			currentTransformMat * transformMat,
			invTransformMat * currentInvTransformMat,
			currentInvTransposedTransformMat* invTransposedTransformMat,
			AABB::transformAABB(referenceMeshAABB, currentTransformMat * transformMat),
			blasInfo.blasRootNodeIndex,
			blasInfo.vertexOffset,
			blasInfo.indexOffset,
			materialID
		};
		instanceDataArray.push_back(instanceData);
	}

	for (const Group& childGroup : group.getChildGroupArray())
	{
		Mat4 transformMat = childGroup.getTransform().getTransformMatrix();
		Mat4 invTransformMat = childGroup.getTransform().getInvTransformMatrix();
		Mat4 invTransposedTransformMat = childGroup.getTransform().getInvTransposeTransformMatrix();
		recursiveBuildInstanceData(instanceDataArray, childGroup, currentTransformMat * transformMat, invTransformMat * currentInvTransformMat, currentInvTransposedTransformMat * invTransposedTransformMat);
	}
}



//=========================================================================================
// TLAS構築のための関数
//=========================================================================================
void Scene::buildTlasBVHNode()
{
	buildTlasBVHNodeRecursively(0, mRayTracingDataOnCPU.instanceDataArray.size());
}

u32 Scene::buildTlasBVHNodeRecursively(const u32 start, const u32 end)
{
	std::vector<BVHNode>& tlasArray = mRayTracingDataOnCPU.tlasArray;
	std::vector<DeviceInstanceData>& instanceDataArray = mRayTracingDataOnCPU.instanceDataArray;


	const u32 nodeIndex = tlasArray.size();
	tlasArray.emplace_back();

	const u32 primitiveCount = end - start;

	// TODO : AABBを親でも計算して、子でも同じ計算をしている。子から渡してもらって合算したほうがいい。
	AABB aabb = AABB::generateAbsolutelyWrappedAABB();
	for (auto iter = instanceDataArray.begin() + start; iter != (instanceDataArray.begin() + end); iter++)
	{
		aabb = AABB::generateWrapingAABB(aabb, iter->aabb);
	}

	tlasArray[nodeIndex].aabb = aabb;

	//------------------------------------------------------------------------------
	// 末端であればここでリターン
	//------------------------------------------------------------------------------
	const u32 maxPrimitiveCount = 4;
	if (primitiveCount <= maxPrimitiveCount)
	{
		tlasArray[nodeIndex].primitiveCount = primitiveCount;
		tlasArray[nodeIndex].firstPrimitiveOffset = start;
		return nodeIndex;
	}

	//------------------------------------------------------------------------------
	// 分割をしていく
	//------------------------------------------------------------------------------
	AABB centroidAABB = AABB::generateAbsolutelyWrappedAABB();
	for (u32 i = start; i < end; i++)
	{
		const Vec3 centroid = (instanceDataArray[i].aabb.getMaxPosition() + instanceDataArray[i].aabb.getMinPosition()) / 2.0f;
		centroidAABB = AABB::generateWrapingAABB(centroidAABB, centroid);
	}

	//------------------------------------------------------------------------------
	// 一番空間的に広がっている軸を探す
	//------------------------------------------------------------------------------
	const u32 splitAxis = centroidAABB.getMostExtendingAxis();
	const f32 splitPoint = (centroidAABB.getMinPosition()[splitAxis] + centroidAABB.getMaxPosition()[splitAxis]) / 2.0f;

	//------------------------------------------------------------------------------
	// プリミティブ配列のソートを行う。
	//------------------------------------------------------------------------------
	auto midiumIter = std::partition(
		instanceDataArray.begin() + start,
		instanceDataArray.begin() + end,
		[splitAxis, splitPoint](const DeviceInstanceData& instanceData)
		{
			const Vec3 centroid = (instanceData.aabb.getMaxPosition() + instanceData.aabb.getMinPosition()) / 2.0f;
			return centroid[splitAxis] < splitPoint;
		}
	);


	u32 midium = std::distance(instanceDataArray.begin(), midiumIter);

	//分割が上手くできなかったとき
	if (midium == start || midium == end)
	{
		midium = start + primitiveCount / 2;
	}

	const u32 leftChildIndex = buildTlasBVHNodeRecursively(start, midium);
	const u32 rightChildIndex = buildTlasBVHNodeRecursively(midium, end);

	tlasArray[nodeIndex].primitiveCount = 0;
	tlasArray[nodeIndex].leftChildOffset = leftChildIndex;
	tlasArray[nodeIndex].rightChildOffset = rightChildIndex;
	
	return nodeIndex;
}
