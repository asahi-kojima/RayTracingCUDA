#pragma once
#include <vector>
#include <map>
#include "Object/hittable.h"
#include "Object/object.h"
#include "Object/mesh.h"
#include "bvh_node.h"
#include "camera.h"


//worldにはデフォルトでプリミティブが定義されていると仮定する。
enum class ObjectType
{
    Default,
    Light
};

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

class WorldRecord
{
public:
    WorldRecord(BvhNode* bvhRootNodeDevicePtr, Object** lightObjectDevicePtrList, u32 lightObjectNum)
    : mBvhRootNodeDevicePtr(bvhRootNodeDevicePtr)
    , mLightObjectDevicePtrList(lightObjectDevicePtrList)
    , mLightObjectNum(lightObjectNum){}

    __device__ __host__ BvhNode* getBvhRootNodeDevicePtr() const {return mBvhRootNodeDevicePtr;}
    __device__ __host__ Object** getLightObjectDevicePtrList() const {return mLightObjectDevicePtrList;}
    __device__ __host__ u32 getLightObjectNum() const {return mLightObjectNum;}
    
private:
    BvhNode* mBvhRootNodeDevicePtr;
    Object** mLightObjectDevicePtrList;
    u32 mLightObjectNum;
};
class World
{
public:
    World();

    void addPrimitive(const std::string& name, Mesh&& primitive);
    void addObject(const char* objectName, const char* primitiveName, const char* materialName, const Transform& transform = Transform(), const SurfaceProperty& surfacePropery = SurfaceProperty());
    void addLightObject(const char* objectName, const char* primitiveName, const char* materialName, const Transform& transform = Transform(), const SurfaceProperty& surfacePropery = SurfaceProperty());
    void setCamera(const Camera& camera);

    void build();
    
    u32 getObjectNum() const;
    u32 getPrimitiveNum() const;
    u32 getMaterialNum() const;
    
    BvhNode* getRootBvhDevicePtr() const;
    
    Camera* getCameraManagedPtr() const;
    
    WorldRecord* getWorldRecordDevicePtr() const;
    
private:
    void buildBvh();


    Camera* mCameraManagedPtr;

    WorldRecord* mWorldRecordManagedPtr;

    // オブジェクトレコードのリスト（GPU上のオブジェクト情報を保持したCPU側のデータ）
    std::map<std::string, ObjectRecord> mString_MapTo_ObjectRecord;

    //Lightオブジェクトのポインタを格納する
    static constexpr u32 MaxLightNum = 10;
    Object* mLightDevicePtrList[MaxLightNum];
    u32 mLightObjectNum;
    Object** mLightObjectManagedList;

    //GPU上のPrimitiveリスト
    std::map<std::string, Primitive*> mString_MapTo_PrimitiveDevPtr;

    //GPU上のマテリアル
    std::map<std::string, Material*> mString_MapTo_MaterialDevPtr;

    //BVHのルートのポインタ
    BvhNode* mRootBvhNodeDevicePtr;


};