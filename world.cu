#include <algorithm>
#include "world.h"

namespace 
{
    //とりあえず100個用意しているが、ここは後ほど精密に行う
    constexpr u32 MaxMaterialNum = 100;
    __managed__ Material* gMaterialList[MaxMaterialNum];
    constexpr u32 MaxPrimitiveNum = 100;
    __managed__ Primitive* gPrimitiveList[MaxPrimitiveNum];
    
    //BVH構築用
    __managed__ BvhNode* rootBvhPtr;
} 

__global__ void setupPrimitives()
{
    gPrimitiveList[0] = new Sphere();
    gPrimitiveList[1] = new AABB();
    gPrimitiveList[2] = new Board();
}

__global__ void setupMaterials()
{
    gMaterialList[0] = new Metal(Color(0xFFFFFF));
    gMaterialList[1] = new Lambertian(Color(0xFFFFFF));
    gMaterialList[2] = new Dielectric(1.33);
    gMaterialList[3] = new Dielectric(1.5);
    gMaterialList[4] = new Dielectric(2.5);
    gMaterialList[5] = new DiffuseLight();
    
}

World::World()
{
    //デフォルトのプリミティブ・マテリアルを登録
    ONCE_ON_GPU(setupPrimitives)();
    ONCE_ON_GPU(setupMaterials)();
    KERNEL_ERROR_CHECKER;

    mString_MapTo_PrimitiveDevPtr["Sphere"] = gPrimitiveList[0];
    mString_MapTo_PrimitiveDevPtr["AABB"] = gPrimitiveList[1];
    mString_MapTo_PrimitiveDevPtr["Board"] = gPrimitiveList[2];
    
    mString_MapTo_MaterialDevPtr["Metal"] = gMaterialList[0];
    mString_MapTo_MaterialDevPtr["Lambert"] = gMaterialList[1];
    mString_MapTo_MaterialDevPtr["Water"] = gMaterialList[2];
    mString_MapTo_MaterialDevPtr["Glass"] = gMaterialList[3];
    mString_MapTo_MaterialDevPtr["Diamond"] = gMaterialList[4];
    mString_MapTo_MaterialDevPtr["DiffuseLight"] = gMaterialList[5];

    //カメラの確保
    CHECK(cudaMallocManaged(&mCameraManagedPtr, sizeof(Camera)));
    CHECK(cudaDeviceSynchronize());
}


//================================================================
// プリミティブメッシュを追加する
//================================================================
void World::addPrimitive(const std::string& name, Mesh&& primitive)
{
    printf("================================================");
    printf("================================================");
    //未実装
    printf("================================================");
    printf("================================================");

    //名前の重複が起きていないかチェック
    if (mString_MapTo_PrimitiveDevPtr.find(name) != mString_MapTo_PrimitiveDevPtr.end())
    {
        printf("already add\n");
        assert(0);
    }

    
    const u32 currentPrimitiveNum = getPrimitiveNum();
    if (currentPrimitiveNum == MaxPrimitiveNum)
    {
        printf("primitive num over\n");
        assert(0);
    }
    Primitive* primitiveDevicePtr = gPrimitiveList[currentPrimitiveNum];
    
    mString_MapTo_PrimitiveDevPtr[name] = primitiveDevicePtr;
}



//=================================================================
// オブジェクトの構築を行う
//=================================================================
__global__ void constructObject(Object* objectPtr, Primitive* primitivePtr, Material* materialPtr, Transform* transformPtr, SurfaceProperty* surfaceProperyPtr)
{
    new (objectPtr) Object(primitivePtr,materialPtr, *transformPtr, *surfaceProperyPtr);
}

void World::addObject(const char* objectName, const char* primitiveName, const char* materialName, const Transform& transform, const SurfaceProperty& surfacePropery)
{
    //------------------------------------------------------
    // 指定されたオブジェクトやプリミティブが正しいかのチェック
    //------------------------------------------------------
    if (auto iter = mString_MapTo_PrimitiveDevPtr.find(primitiveName); iter == mString_MapTo_PrimitiveDevPtr.end())
    {
        printf("specified pritmitive name does not exist in Map\n");
        exit(1);
    }
    if (auto iter = mString_MapTo_MaterialDevPtr.find(materialName); iter == mString_MapTo_MaterialDevPtr.end())
    {
        printf("specified material name does not exist in Map\n");
        exit(1);
    }
    if (auto iter = mString_MapTo_ObjectRecord.find(objectName); iter != mString_MapTo_ObjectRecord.end())
    {
        printf("specified object name already exist in Map : %s\n", iter->second.getObjectName());
        exit(1);
    } 

    //---------------------------------------------------------------------
    //指定されたプリミティブとマテリアルの名前から、GPU上のアドレスを準備する。
    //---------------------------------------------------------------------
    Primitive* primitivePtrD = mString_MapTo_PrimitiveDevPtr[primitiveName];
    Material* materialPtrD = mString_MapTo_MaterialDevPtr[materialName];

    //--------------------------------------------------
    //GPU上にトランスフォームを用意しておく
    //--------------------------------------------------
    Transform* transformPtrD = nullptr;
    CHECK(cudaMalloc(&transformPtrD, sizeof(Transform)));
    CHECK(cudaMemcpy(transformPtrD, &transform, sizeof(Transform), cudaMemcpyHostToDevice));

    //--------------------------------------------------
    //GPU上に表面プロパティを用意しておく
    //--------------------------------------------------
    SurfaceProperty* surfaceProperyDevicePtr = nullptr;
    CHECK(cudaMalloc(&surfaceProperyDevicePtr, sizeof(SurfaceProperty)));
    CHECK(cudaMemcpy(surfaceProperyDevicePtr, &surfacePropery, sizeof(SurfaceProperty), cudaMemcpyHostToDevice));
    
    //--------------------------------------------------
    //GPU上でオブジェクトインスタンスをメモリ上に構築
    //--------------------------------------------------
    Object* objectPtrD = nullptr;
    CHECK(cudaMalloc(&objectPtrD, sizeof(Object)));
    ONCE_ON_GPU(constructObject)(objectPtrD, primitivePtrD,materialPtrD, transformPtrD, surfaceProperyDevicePtr);
    KERNEL_ERROR_CHECKER;

    //-----------------------------------------------------
    // GPU上に構築したオブジェクト情報を参照する為のレコード
    //-----------------------------------------------------
    ObjectRecord record(objectPtrD, objectName, primitiveName, materialName, transform, transformPtrD);
    mString_MapTo_ObjectRecord[objectName] = record;
}


//=================================================================
// カメラをセットする
//=================================================================
void World::setCamera(const Camera& camera)
{
    *mCameraManagedPtr = camera;
}


//========================================================================================
// BVHの構築
//========================================================================================
void sort_along_axis(std::vector<std::pair<Vec3, Object*>>& pairs, const u32 start, u32 end, u32 depth = 0)
{
    if (end - start == 1)
    {
        return;
    }

    std::sort(
        pairs.begin() + start, 
        pairs.begin() + end, 
        [depth](std::pair<Vec3, Object*>& pair0, std::pair<Vec3, Object*> pair1) {const u32 axis_of_sort = depth % 3; return pair0.first[axis_of_sort] < pair1.first[axis_of_sort]; });

    sort_along_axis(pairs, start, start + (end - start) / 2, depth + 1);
    sort_along_axis(pairs, start + (end - start) / 2, end, depth + 1);
}


__device__ BvhNode* recursiveBuildBvh(const u32 start, const u32 range, BvhNode nodeArray[], Object* objectPtrList[], u32& memoryId)
{
    const u32 index = memoryId;
    memoryId++;
    //葉ノードに到着したので、オブジェクトを登録
    if (range == 1)
    {
        new (nodeArray + index) BvhNode(objectPtrList[start], objectPtrList[start]->getAABB());
        return nodeArray + index;
    }

    BvhNode* lhs = recursiveBuildBvh(start,              range / 2,      nodeArray, objectPtrList, memoryId);
    BvhNode* rhs = recursiveBuildBvh(start + range / 2, (range + 1) / 2, nodeArray, objectPtrList, memoryId);


    AABB aabb = AABB::wraping(lhs->getAABB(), rhs->getAABB());
    new (nodeArray + index) BvhNode(lhs, rhs, aabb);

    return nodeArray + index;
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
__global__ void buildBvhOnDevice(const u32 objectNum, Object* objectPtrList[])
{
    //ノードがいくつ必要か計算する
    const u32 nodeNum = 2 * objectNum - 1;

    //メモリ上に一列にノードを確保する
    BvhNode* bvhArray = reinterpret_cast<BvhNode*>(malloc(sizeof(BvhNode) * nodeNum));

    {
        BvhNode* bvhNodePtr;
        u32 memoryId = 0;
        BvhNode* rootNodePtr =  recursiveBuildBvh(0, objectNum, bvhArray, objectPtrList, memoryId);

        rootBvhPtr = rootNodePtr;
    }
}    


void World::buildBvh()
{
    if (getObjectNum() == 0)
    {
        assert(0);
        printf("Object Num must be more than 1\n");
        exit(1);
    }

    //--------------------------------------------------------------------
    // BVHを構築する前にCPU側でソートをしておく。
    //--------------------------------------------------------------------
    std::vector<std::pair<Vec3, Object*>> mPair_Translation_ObjectDevicePtr_List;
    for (auto iter = mString_MapTo_ObjectRecord.begin(), end = mString_MapTo_ObjectRecord.end(); iter != end; iter++)
    {
        const Vec3& translation = iter->second.getTransform().getTranslation();
        Object* objectDevicePtr = iter->second.getObjectDevicePtr();
        mPair_Translation_ObjectDevicePtr_List.push_back(std::make_pair(translation, objectDevicePtr));
        //printf("%f, %f, %f\n", translation[0], translation[1], translation[2]);
    }
    #ifdef DEBUG
        if (mPair_Translation_ObjectDevicePtr_List.size() != getObjectNum())
        {
            printf("size mismatch\n");
            assert(false);
        }
    #endif
    sort_along_axis(mPair_Translation_ObjectDevicePtr_List, 0, getObjectNum());

    //---------------------------------------------------------------------
    // オブジェクトのトランスフォームを基準にソート
    //---------------------------------------------------------------------
    BvhNode** bvhPtrList;
    CHECK(cudaMalloc(&bvhPtrList, sizeof(BvhNode*) * getObjectNum()));

    Object** objectDevicePtrList;//Object* array[size]; 
    CHECK(cudaMallocManaged(&objectDevicePtrList, sizeof(Object*) * getObjectNum())); CHECK(cudaDeviceSynchronize());
    for (auto iter = mPair_Translation_ObjectDevicePtr_List.begin(), begin = mPair_Translation_ObjectDevicePtr_List.begin(), end = mPair_Translation_ObjectDevicePtr_List.end(); iter != end; iter++)
    {
        const u32 index = (iter - begin);
        objectDevicePtrList[index] = iter->second;
    }

    //---------------------------------------------------------------------
    // オブジェクトのBVHを構築する
    //---------------------------------------------------------------------
    ONCE_ON_GPU(buildBvhOnDevice)(getObjectNum(), objectDevicePtrList);
    CHECK(cudaDeviceSynchronize());

    mRootBvhNodeDevicePtr = rootBvhPtr;
}



//============================================================================================
//============================================================================================

u32 World::getObjectNum() const
{
    return mString_MapTo_ObjectRecord.size();
}

u32 World::getPrimitiveNum() const
{
    return mString_MapTo_PrimitiveDevPtr.size();
}

u32 World::getMaterialNum() const
{
    return mString_MapTo_MaterialDevPtr.size();
}

BvhNode* World::getRootBvhDevicePtr() const
{
    return mRootBvhNodeDevicePtr;
}

Camera* World::getCameraManagedPtr() const
{
    return mCameraManagedPtr;
}