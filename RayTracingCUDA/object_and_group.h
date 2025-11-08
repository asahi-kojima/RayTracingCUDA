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
	void setTransform(const Transform& transform);
	

	const std::string& getName() const { return mName; }
	Transform getTransform() const;
	Transform& getTransform();


	Result addChildObject(const Object& object, const Transform& transform = Transform::identity());
	Result addChildGroup(const Group& group, const Transform& transform = Transform::identity());

	void getAllMeshNamesChildrenHave(std::vector<std::string>& nameArray) const;
	void getAllMateialNamesChildrenHave(std::vector<std::string>& nameArray) const;

private:
	std::string mName; // Groupの名前
	Transform mTransform; // このGroup自体のTransform
	
	std::vector<Group> mChildGroupArray;
	std::vector<Object> mChildObjectArray;

	bool doThisNameExistAtTopLevel(const std::string& name) const;
};