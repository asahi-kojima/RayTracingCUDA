#include <format>
#include "object_and_group.h"

Object::Object(const std::string& objectName, const std::string& meshName, const std::string& materialName, const Transform& transform)
	: mName(objectName)
	, mRefMeshName(meshName)
	, mRefMaterialName(materialName)
	, mTransform(transform)
{
}

void Object::updateTransform()
{
	mTransform.updateTransformMatrices();
}


Group::Group(const std::string& name, const Transform& transform)
	: mName(name)
	, mTransform(transform)
	, mChildGroupArray()
	, mChildObjectArray()
{
}


Result Group::addChildObject(const Object& object, const Transform& transform, const std::string& newName)
{
	Object tmpObject = object;
	tmpObject.setTransform(transform);

	if (newName != std::string(""))
	{
		tmpObject.setName(newName);
	}

	//既にグループ内の同じ階層に同名のオブジェクトかグループがないか確認し、
	//あれば名前に通し番号を付ける: object -> object_1みたいに。
	if (doThisNameExistAtTopLevel(tmpObject.getName()))
	{
		u32 tryCount = 1;
		const u32 maxTryCount = 10;
		while (true)
		{
			std::string newNameCand = tmpObject.getName() + "_" + std::to_string(tryCount);
			if (!doThisNameExistAtTopLevel(newNameCand))
			{
				tmpObject.setName(newNameCand);
				break;
			}

			tryCount++;
			if (tryCount == maxTryCount)
			{
				assert(0);
				return Result{ false, std::format("There are many names similar to Specified Name [%s].", object.getName().c_str()) };
			}
		}
	}

	mChildObjectArray.push_back(tmpObject);
	return true;
}

Result Group::addChildGroup(const Group& group, const Transform& transform, const std::string& newName)
{
	Group tmpGroup = group;
	tmpGroup.setTransform(transform);

	if (newName != std::string(""))
	{
		tmpGroup.setName(newName);
	}

	//既にグループ内の同じ階層に同名のオブジェクトかグループがないか確認し、
	//あれば名前に通し番号を付ける: object -> object_1みたいに。
	if (doThisNameExistAtTopLevel(tmpGroup.getName()))
	{
		u32 tryCount = 1;
		const u32 maxTryCount = 10;
		while (true)
		{
			std::string newNameCand = tmpGroup.getName() + "_" + std::to_string(tryCount);
			if (!doThisNameExistAtTopLevel(newNameCand))
			{
				tmpGroup.setName(newNameCand);
				break;
			}

			tryCount++;
			if (tryCount == maxTryCount)
			{
				assert(0);
				return Result{false, std::format("There are many names similar to Specified Name [%s].", group.getName().c_str())};
			}
		}
	}

	mChildGroupArray.push_back(tmpGroup);
	return true;
}

u32 Group::getDescendantObjectCount() const
{
	u32 childCount = mChildObjectArray.size();
	for (const Group& childGroup : mChildGroupArray)
	{
		childCount += childGroup.getDescendantObjectCount();
	}

	return childCount;
}

void Group::getAllMeshNamesChildrenHave(std::vector<std::string>& nameArray) const
{
	for (const auto& object : mChildObjectArray)
	{
		nameArray.push_back(object.getMeshName());
	}

	for (const auto& group : mChildGroupArray)
	{
		group.getAllMeshNamesChildrenHave(nameArray);
	}
}

void Group::getAllMateialNamesChildrenHave(std::vector<std::string>& nameArray) const
{
	for (const auto& object : mChildObjectArray)
	{
		nameArray.push_back(object.getMaterialName());
	}

	for (const auto& group : mChildGroupArray)
	{
		group.getAllMateialNamesChildrenHave(nameArray);
	}
}

void Group::updateAllChildrenTransform()
{
	mTransform.updateTransformMatrices();

	for (auto& object : mChildObjectArray)
	{
		object.updateTransform();
	}

	for (auto& group : mChildGroupArray)
	{
		group.updateAllChildrenTransform();
	}
}

bool Group::doThisNameExistAtTopLevel(const std::string& name) const
{
	for (const auto& object : mChildObjectArray)
	{
		if (object.getName() == name)
		{
			return true;
		}
	}

	for (const auto& group : mChildGroupArray)
	{
		if (group.getName() == name)
		{
			return true;
		}
	}

	return false;
}



