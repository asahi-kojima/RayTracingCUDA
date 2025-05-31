#pragma once

#include <vector>
#include <memory>
#include <string>

#include "vector.h"
#include "typeinfo.h"
#include "render-target.h"
#include "camera.h"
#include "hittable.h"
#include "node.h"
#include "world.h"

class RayTracingEngine
{
public:
	__host__ RayTracingEngine();
	__host__ ~RayTracingEngine();

	//描画するワールド（シーン）をセット
	__host__ void setWorld(World& world);

	//レンダーターゲットをセット
	__host__ void setRenderTarget(RenderTarget& target);

	//セットしたワールドとレンダーターゲットに対しレンダリング
	__host__ void render(const u32 sampleSize = 30, const u32 depth = 15);

	//レンダリング結果を指定パスに保存
	__host__ void saveRenderResult(const std::string& path);


private:
	RenderTarget mRenderTarget;
	Node* mRootNode = nullptr;

	World* mWorldPtr;
};