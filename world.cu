#include <algorithm>
#include "world.h"


//とりあえず100個用意しているが、ここは後ほど精密に行う
constexpr u32 MaxMaterialNum = 100;
__managed__ Material* gMaterialList[MaxMaterialNum];
constexpr u32 MaxPrimitiveNum = 100;
__managed__ Primitive* gPrimitiveList[MaxPrimitiveNum];

__global__ void setupPrimitives()
{
    gPrimitiveList[0] = new Sphere();
    gPrimitiveList[1] = new AABB();
}

__global__ void setupMaterials()
{
    gMaterialList[0] = new Metal(Color(0xFF0000));
    gMaterialList[1] = new Lambertian(Color(0x0000FF));
}

World::World()
{

    ONCE_ON_GPU(setupPrimitives)();
    ONCE_ON_GPU(setupMaterials)();
    KERNEL_ERROR_CHECKER;

    mString_MapTo_PrimitiveDevPtr["Sphere"] = gPrimitiveList[0];
    mString_MapTo_PrimitiveDevPtr["AABB"] = gPrimitiveList[1];
    
    mString_MapTo_MaterialDevPtr["Metal"] = gMaterialList[0];
    mString_MapTo_MaterialDevPtr["Lambert"] = gMaterialList[1];
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
__global__ void constructObject(Object* objectPtr, Primitive* primitivePtr, Material* materialPtr, Transform* transformPtr)
{
    new (objectPtr) Object(primitivePtr,materialPtr, *transformPtr);
}

void World::addObject(const char* objectName, const char* primitiveName, const char* materialName, const Transform& transform)
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
    
    // 今後、マネージドメモリから初期化を行えるようにする
    // CHECK(cudaMallocManaged(&objectPtrOnGpu, sizeof(Object)));
    // CHECK(cudaDeviceSynchronize());
    // // new (objectPtrOnGpu) Object(primitivePtr_d, materialPtr_d, transform);
    // new (objectPtrOnGpu) Object(primitivePtr_d, materialPtr_d);
    // //GPU_ERROR_CHECKER(cudaGetLastError());
    
    //--------------------------------------------------
    //GPU上でオブジェクトインスタンスをメモリ上に構築
    //--------------------------------------------------
    Object* objectPtrD = nullptr;
    CHECK(cudaMalloc(&objectPtrD, sizeof(Object)));
    ONCE_ON_GPU(constructObject)(objectPtrD, primitivePtrD,materialPtrD, transformPtrD);
    KERNEL_ERROR_CHECKER;


    //-----------------------------------------------------
    // GPU上に構築したオブジェクト情報を参照する為のレコード
    //-----------------------------------------------------
    ObjectRecord record(objectPtrD, objectName, primitiveName, materialName, transform, transformPtrD);
    mString_MapTo_ObjectRecord[objectName] = record;
}


//========================================================================================
// BVHの構築
//========================================================================================
__device__ void makeLeafNode()
{

}

__global__ void buildBvhOnGPU(u32 objectNum)
{
    // //ノードがいくつ必要か計算する
    // const u32 nodeNum = 2 * objectNum - 1;

    // //メモリ上に一列にノードを確保する
    // BvhNode* bvhArray = new BvhNode[nodeNum];

    // makeLeafNode();
    // //makeBvhNode();
}

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


void World::buildBvh()
{
    if (getObjectNum() == 0)
    {
        assert(0);
    }

    std::vector<std::pair<Vec3, Object*>> mPair_Translation_ObjectDevicePtr_List;
    for (auto iter = mString_MapTo_ObjectRecord.begin(), end = mString_MapTo_ObjectRecord.end(); iter != end; iter++)
    {
        const Vec3& translation = iter->second.getTransform().getTranslation();
        Object* objectDevicePtr = iter->second.getObjectDevicePtr();
        mPair_Translation_ObjectDevicePtr_List.push_back(std::make_pair(translation, objectDevicePtr));
        printf("%f, %f, %f\n", translation[0], translation[1], translation[2]);
    }
    sort_along_axis(mPair_Translation_ObjectDevicePtr_List, 0, getObjectNum());

#ifdef DEBUG
    if (mPair_Translation_ObjectDevicePtr_List.size() != getObjectNum())
    {
        printf("size mismatch\n");
        assert(false);
    }
#endif
    ONCE_ON_GPU(buildBvhOnGPU)(getObjectNum());
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