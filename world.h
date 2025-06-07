#pragma once
#include <vector>
#include <map>
#include "Object/hittable.h"
#include "Object/object.h"
#include "Object/mesh.h"
#include "bvh_node.h"
#include "camera.h"


//worldにはデフォルトでプリミティブが定義されていると仮定する。


struct ObjectRecord
{
public:
    ObjectRecord() = default;
    ObjectRecord(Object* objectPtrOnGpu, const char* objectName, const char* primitiveName, const char* materialName, const Transform& transform, Transform* transformPtrD)
    : mObjectDevicePtr(objectPtrOnGpu)
    , mObjectName(objectName)
    , mPrimitiveName(primitiveName)
    , mMaterialName(materialName)
    , mTransform(transform)
    , mTransformPtrD(transformPtrD)
    {}

    const char* getObjectName() const
    {
        return mObjectName;
    }

    const Transform& getTransform() const
    {
        return mTransform;
    }

    Object* getObjectDevicePtr() const
    {
        return mObjectDevicePtr;
    }

private:
    Object* mObjectDevicePtr;
    const char* mObjectName;

    const char* mPrimitiveName;
    
    const char* mMaterialName;
    
    Transform mTransform;
    Transform* mTransformPtrD;
};

class World
{
public:
    World();

    void addPrimitive(const std::string& name, Mesh&& primitive);
    void addObject(const char* objectName, const char* primitiveName, const char* materialName, const Transform& transform = Transform(), const SurfaceProperty& surfacePropery = SurfaceProperty());
    void setCamera(const Camera& camera);
    void buildBvh();

    u32 getObjectNum() const;
    u32 getPrimitiveNum() const;
    u32 getMaterialNum() const;

    BvhNode* getRootBvhDevicePtr() const;

    Camera* getCameraManagedPtr() const;

private:
    Camera* mCameraManagedPtr;

    // オブジェクトレコードのリスト（GPU上のオブジェクト情報を保持したCPU側のデータ）
    std::map<std::string, ObjectRecord> mString_MapTo_ObjectRecord;

    //GPU上のPrimitiveリスト
    std::map<std::string, Primitive*> mString_MapTo_PrimitiveDevPtr;

    //GPU上のマテリアル
    std::map<std::string, Material*> mString_MapTo_MaterialDevPtr;

    //BVHのルートのポインタ
    BvhNode* mRootBvhNodeDevicePtr;
};