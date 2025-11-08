#include <format>
#include "scene.h"

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

Result Scene::addObject(const Object& object, const Transform& transform)
{
	//objectの追加時に、MeshとMaterialが存在している正規の物か確認する
	const std::string& meshName = object.getMeshName();
	if (mMeshNameToIdMap.count(meshName) == 0)
	{
		return Result(false, std::format("Specified mesh [%s] does not exist in the scene.", meshName.c_str()));
	}

	const std::string& materialName = object.getMaterialName();
	if (mMaterialNameToIdMap.count(materialName) == 0)
	{
		return Result(false, std::format("Specified Material [%s] does not exist in the scene.", materialName.c_str()));

	}

	//確認が取れたら登録する
	Result result = mRootGroup.addChildObject(object, transform);

	return result;
}

Result Scene::addGroup(const Group& group, const Transform& transform)
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
	Result result = mRootGroup.addChildGroup(group, transform);

	return result;
}