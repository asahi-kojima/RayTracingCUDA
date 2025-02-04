#pragma once

//#include "debug-setting.h"
//#include "object.h"
//#include "camera.h"
//#include "node.h"
//#include "render-target.h"
//#include "util.h"
//#include "thread-pool.h"
#include "vector.h"
#include "typeinfo.h"
#include "render-target.h"
#include "camera.h"
#include "hittable.h"
#include "node.h"
#include <vector>
#include <memory>
#include <string>
struct SecondaryInfoByRay
{
	u32 depth;
};


class RayTracingEngine
{
public:
	__host__ RayTracingEngine();
	~RayTracingEngine();

	__host__ void setObjects(const std::vector<Hittable*>& world);

	__host__ void setCamera(const Camera& camera);

	__host__ void setRenderTarget(RenderTarget& target);

	__host__ void render(const u32 sampleSize = 30, const u32 depth = 15);

	__host__ void saveRenderResult(const std::string& path);

	//void drawTrajectory(u32 i, u32 j);

private:
	CameraCore* mCamera;
	RenderTarget mRenderTarget;
	Node* mRootNode = nullptr;


};