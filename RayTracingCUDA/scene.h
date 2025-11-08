#pragma once
#include <unordered_map>
#include "mesh.h"
#include "material.h"
#include "object_and_group.h"

struct DeviceInstanceData
{

};

struct DeviceRequiredData
{
	std::vector<Vertex> vertexArray;
	std::vector<u32> indexArray;

	std::vector<DeviceInstanceData> instanceDataArray;

	std::vector<Material> materialArray;
};

class Scene
{
public:
	Scene()
		: mMeshArray(0)
		, mMaterialArray(0)
		//, mInstanceArray(0)
		, mRootGroup("root")
	{ }

	s32 addMesh(const std::string& meshName ,const Mesh& mesh);
	s32 addMaterial(const std::string& materialName, const Material& material);

	Result addObject(const Object& object, const Transform& transform = Transform::identity());
	Result addGroup(const Group& group, const Transform& transform = Transform::identity());

private:
	std::vector<Mesh> mMeshArray;
	std::vector<Material> mMaterialArray;

	std::unordered_map<std::string, u32> mMeshNameToIdMap;
	std::unordered_map<std::string, u32> mMaterialNameToIdMap;

	Group mRootGroup;
};