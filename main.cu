#include <stdio.h>
#include <vector>
#include <memory>
#include <vector>
#include <string>
#include <curand_kernel.h>
#include "vector.h"
#include "color.h"
#include "ray.h"
#include "object.h"
#include "util.h"
#include "hittable.h"
#include "texture.h"
#include "engine.h"

// /usr/local/cuda/bin/nvcc --generate-code arch=compute_86,code=sm_86 -std=c++17 -rdc=true -O3 -DNDEBUG -w *.cu && ./a.out && convert ./build/result.ppm ./build/rayTracingDemo.png

__device__ curandState s[32];

__global__ void setup_gpu()
{
	for (u32 i = 0; i < 32; i++)
	{
		curand_init(static_cast<unsigned long long>(i), 0, 0, &s[i]);
	}
}

int main(int argc, char** argv)
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 100);
	setup_gpu<<<1, 1>>>();
	//=================================================================
	// オブジェクトの準備
	//=================================================================
	World world;
	world.addObject();


	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 1;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	Vec3 lookAt(0,0,0);
	// Vec3 lookFrom(0.5,0.2, 1);
	// Vec3 lookFrom(0.9, 0.4, 1);
	Vec3 lookFrom(1.0, 5, 3.0);
	lookFrom *= (20 / lookFrom.length());
	Camera camera = Camera(lookFrom, lookAt, Vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());

	//=================================================================
	// レンダーターゲットの準備
	//=================================================================
	RenderTarget renderTarget[3] = {RenderTarget(resolutionX, resolutionY), RenderTarget(resolutionX, resolutionY), RenderTarget(resolutionX, resolutionY)};

	//=================================================================
	// オブジェクトの準備
	//=================================================================
	RayTracingEngine engine;

	engine.setObjects(world);
	engine.setRenderTarget(renderTarget[0]);

	camera = Camera(lookFrom, lookAt, Vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, 2 * (lookFrom - lookAt).length());
	engine.setCamera(camera);
	engine.render(30, 50);

	std::string s = "./build/result";
	s += std::to_string(0);
	s += ".ppm";
	engine.saveRenderResult(s);
}
