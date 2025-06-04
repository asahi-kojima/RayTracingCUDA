#pragma once
#include <vector>
#include "hittable.h"
#include "bvh_node.h"
//worldにはデフォルトでプリミティブが定義されていると仮定する。

class World
{
public:
    void addObject();

    void buildWorldOnGpu();

    u32 getObjectNum() const;

    //ワールドをGPU上に構築する
    void build();

private:
    // オブジェクトのリスト
    std::vector<Hittable*> mHittableList;

    // 
    BvhNode* mBvhRootNode;
    
};