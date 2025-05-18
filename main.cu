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

	constexpr f32 Scale = 0.3;

	constexpr s32 Num = 10;
	constexpr f32 Range = 1.0f;
	constexpr f32 Diff = Range * Scale / Num;
	for (s32 xid = -Num; xid <= Num; xid++)
	{
		for (s32 yid = -3; yid <= 0; yid++)
		{
			for (s32 zid = -Num; zid <= Num; zid++)
			{
				const f32 x = Diff * xid;
				const f32 y = Diff * yid;
				const f32 z = Diff * zid;

				const vec3 pos(x, y, z);

				f32 scale = Diff * 0.3;
				vec3 extension = vec3::one() * Diff / 2;
				extension += vec3(0, RandomGenerator::uniform_real() * Scale, 0);

				Material *material = make_material<Metal>(Color::Bronze);
				if (RandomGenerator::uniform_real() < 0.3)
				{
					material = make_material<Metal>(Color::Blue);
				}
				if (RandomGenerator::uniform_real() < 0.1)
				{
					f32 dx[4] = {Diff / 4,Diff / 4,-Diff / 4,-Diff / 4};
					f32 dz[4] = {Diff / 4,-Diff / 4,Diff / 4,-Diff / 4};
					for (s32 i = 0; i < 4; i++)
					{
						const vec3 pos(x + dx[i], y, z + dz[i]);
						vec3 extension = vec3::one() * Diff / 4;
						extension += vec3(0, RandomGenerator::uniform_real() * Scale, 0);
						material = make_material<Metal>(Color::Silver);
						world.push_back(make_object<AABB>(pos - extension, pos + extension, material));
					}
				}
				else
				{	
					world.push_back(make_object<AABB>(pos - extension, pos + extension, material));
				}
			}
		}
	}

	// vec3 origin(0, 0, 0);
	// world.push_back(make_object<Sphere>(origin, 0.2 * max_radius,make_material<QuasiGravitationalField2>(0.10, origin)));

	// vec3 origin(10, 10, 10);
	// vec3 extension(3, 3, 3);
	// world.push_back(make_object<AABB>(origin - extension, origin + extension,make_material<SunLight>(10.0f)));

	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 1;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(-1, 0, -1);
	lookAt *= (Range * Scale);
	// vec3 lookFrom(13, 2, 5);
	vec3 lookFrom(0.9, 1.5, 1.1f);
	lookFrom *= (Range * Scale * 1.2);
	// lookFrom *= (0.9 / lookFrom.length());

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
