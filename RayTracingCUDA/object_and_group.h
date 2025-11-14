#pragma once
#include "transform.h"

/// <summary>
/// シーンに配置するオブジェクトの情報を持ったクラス
/// メッシュやマテリアルへの参照のみを持つため軽量
/// </summary>
class Object
{
public:
	Object(const std::string& objectName, const std::string& meshName, const std::string& materialName, const Transform& transform = Transform::identity());

	void setName(const std::string& objectName) { mName = objectName; }
	void setMesh(const std::string& meshName) { mRefMeshName = meshName; }
	void setMaterial(const std::string& materialName) { mRefMaterialName = materialName; }
	void setTransform(const Transform& transform) { mTransform = transform; }

	const std::string& getName() const { return mName; }
	const std::string& getMeshName() const { return mRefMeshName; }
	const std::string& getMaterialName() const { return mRefMaterialName; }
	const Transform& getTransform() const { return mTransform; }

	void updateTransform();

private:
	std::string mName;
	std::string mRefMeshName;
	std::string mRefMaterialName;
	Transform mTransform;
};

/// <summary>
/// オブジェクトを纏める為のクラス
/// </summary>
class Group
{
public:
	Group(const std::string& name, const Transform& transform = Transform::identity());
	~Group() = default;


	void setName(const std::string& name) { mName = name; }
	void setTransform(const Transform& transform) { mTransform = transform; }
	

	const std::string& getName() const { return mName; }
	const Transform& getTransform() const { return mTransform; }


	Result addChildObject(const Object& object, const Transform& transform = Transform::identity(), const std::string& newName = std::string(""));
	Result addChildGroup(const Group& group, const Transform& transform = Transform::identity(), const std::string& newName = std::string(""));

	const std::vector<Group>& getChildGroupArray() const { return mChildGroupArray; }
	const std::vector<Object>& getChildObjectArray() const { return mChildObjectArray; }

	u32 getChildGroupCount() const { return mChildGroupArray.size(); }
	u32 getChildObjectCount() const { return mChildObjectArray.size(); }
	u32 getDescendantObjectCount() const;

	void getAllMeshNamesChildrenHave(std::vector<std::string>& nameArray) const;
	void getAllMateialNamesChildrenHave(std::vector<std::string>& nameArray) const;

	void updateAllDescendantsTransform();
	//void flatten(std::vector<f32>& flattenDataArray, const Transform& transform) const;

private:
	std::string mName; // Groupの名前
	Transform mTransform; // このGroup自体のTransform
	
	std::vector<Group> mChildGroupArray; // 子グループ
	std::vector<Object> mChildObjectArray; // 子オブジェクト

	bool doThisNameExistAtTopLevel(const std::string& name) const;
};