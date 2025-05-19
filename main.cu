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

int main()
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 100);
	setup_gpu<<<1, 1>>>();
	//=================================================================
	// オブジェクトの準備
	//=================================================================
	std::vector<Hittable *> world;

	const f32 Scale = 3;
	const f32 TriangleScale = 1.1;
	for (u32 i = 0; i < 1000; i++)
	{
		vec3 center = vec3::signed_uniform_real_vector() * Scale;
		const vec3 v0 = center + vec3::signed_uniform_real_vector() * TriangleScale;
		const vec3 v1 = center + vec3::signed_uniform_real_vector() * TriangleScale;
		const vec3 v2 = center + vec3::signed_uniform_real_vector() * TriangleScale;
		world.push_back(make_object<Triangle>(v0, v1, v2, make_material<Metal>(Color(RandomGenerator::uniform_int(0, 0xFFFFFF))), false));
	}


	// world.push_back(make_object<Sphere>(vec3(1,0, -1), 0.5, make_material<Metal>(Color::Blue)));
	// world.push_back(make_object<Sphere>(vec3(-1, 0, -1), 0.5, make_material<Metal>(Color(0.8, 0.8, 0.8))));
	// world.push_back(make_object<Sphere>(vec3(-1, 0, -1), 0.5, make_material<Metal>(Color(0.8, 0.8, 0.8))));
	world.push_back(make_object<Sphere>(vec3(0, 0, 0), 0.4, make_material<Metal>(Color(0.8, 0.8, 0.8))));





	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 1;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(0,0,0);
	vec3 lookFrom(-0, 0, 10);

	Camera camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());

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

	camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, 2 * (lookFrom - lookAt).length());
	engine.setCamera(camera);
	engine.render(10, 50);

	std::string s = "./build/result";
	s += std::to_string(0);
	s += ".ppm";
	engine.saveRenderResult(s);
}
