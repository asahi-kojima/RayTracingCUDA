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

	constexpr s32 Range = 10;
	for (s32 w = -Range; w <= Range; w+= 1)
	{
		for (s32 h = -Range; h <= Range; h+=1)
		{
			for (s32 z = 2; z <= 15; z++)
			{
				f32 which = RandomGenerator::uniform_real();
				const f32 scale = 1.0f;
				const vec3 pos = vec3(w , h  , -z) * scale;

				Material* material = nullptr;

				// if (h == 0 && w == 0 && z == 0)
				// {
				// 	//material = make_material<GravitationalField>(1.2, pos);
				// 	//world.push_back(make_object<AABB>(vec3(-0.1, -0.1, -0.1),vec3(0.1, 0.1, 0.1), material));
				// 	continue;
				// }
				// else if (h == 0 && w == 0)
				// {
				// 	continue;
				// }
				// else if (h == 0 || w == 0)
				// {
				// 	continue;
				// }
				// // else if (h == 0 && w == 0)
				// // {
				// // 	material = make_material<Rutherford>(10.5, pos);
				// // }
				// else
				{
					
					material = make_material<Metal>(Color(RandomGenerator::uniform_int(0, 0xFFFFFF)));
					vec3 diff_x(RandomGenerator::uniform_real(),RandomGenerator::uniform_real(),RandomGenerator::uniform_real());
					vec3 diff_y(RandomGenerator::uniform_real(),RandomGenerator::uniform_real(),RandomGenerator::uniform_real());
					f32 diff_scale = 0.1f;

					// world.push_back(make_object<AABB>(
					// 	pos + vec3(-0.1, -0.1, -0.1)*2,
					// 	pos + vec3(0.1, 0.1, 0.1)*2, 
					// 	material));
					world.push_back(make_object<Sphere>(pos,0.2, material));

					continue;
				}
				
				world.push_back(make_object<Sphere>(pos, 0.1f, material));
			}
		}
	}

	{
		vec3 center(0,0,2);
		vec3 extention = vec3::one() * 0.1;
		//world.push_back(make_object<AABB>(center - extention, center + extention,make_material<Metal>(Color::White)));
	}
	// world.push_back(make_object<Sphere>(vec3(1,0, -1), 0.5, make_material<Metal>(Color::Blue)));
	// world.push_back(make_object<Sphere>(vec3(-1, 0, -1), 0.5, make_material<Metal>(Color(0.8, 0.8, 0.8))));
	// world.push_back(make_object<Sphere>(vec3(-1, 0, -1), 0.5, make_material<Metal>(Color(0.8, 0.8, 0.8))));
	//world.push_back(make_object<Sphere>(vec3(0, 0, 0), 1, make_material<Dielectric>(1.5)));
	{
		Material* material = make_material<Dielectric>(1.2f);
		//material = make_material<Lambertian>(Color::Bronze);
		vec3 center(0,0,0);
		vec3 extention = vec3::one() * 1;
#if 1
		world.push_back(make_object<AABB>(center - extention, center + extention,material));
#else
		vec3 v0(+1, +1, +1);
		vec3 v1(+1, -1, +1);
		vec3 v2(-1, +1, +1);
		vec3 v3(-1, -1, +1);
		vec3 v4(+1, +1, -1);
		vec3 v5(+1, -1, -1);
		vec3 v6(-1, +1, -1);
		vec3 v7(-1, -1, -1);
		world.push_back(make_object<Triangle>(v3, v0, v2 , material));
		world.push_back(make_object<Triangle>(v3, v1, v0 , material));

		world.push_back(make_object<Triangle>(v1, v4, v0 , material));
		world.push_back(make_object<Triangle>(v1, v5, v4 , material));

		world.push_back(make_object<Triangle>(v7, v2, v6 , material));
		world.push_back(make_object<Triangle>(v7, v3, v2 , material));

		world.push_back(make_object<Triangle>(v5, v6, v4 , material));
		world.push_back(make_object<Triangle>(v5, v7, v6 , material));

		world.push_back(make_object<Triangle>(v2, v4, v6 , material));
		world.push_back(make_object<Triangle>(v2, v0, v4 , material));

		world.push_back(make_object<Triangle>(v7, v1, v3 , material));
		world.push_back(make_object<Triangle>(v7, v5, v1 , material));
#endif
	}
	{
		vec3 v0(0, 0, 0);
		vec3 v1(0.1, 0.1, 0);
		vec3 v2(0, 0.1, 0);
		//world.push_back(make_object<Triangle>(v0, v1, v2,make_material<Dielectric>(1.5)));
	}

	// vec3 center(5, 0, 5);
	// vec3 extention(1,1,1);
	// world.push_back(make_object<AABB>(center - extention, center + extention,make_material<SunLight>(Color::White, 30)));


	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 4;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(0,0,0);
	// vec3 lookFrom(0.5,0.2, 1);
	// vec3 lookFrom(0.9, 0.4, 1);
	vec3 lookFrom(8.0, 0.4, 8);

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
	engine.render(1, 50);

	std::string s = "./build/result";
	s += std::to_string(0);
	s += ".ppm";
	engine.saveRenderResult(s);
}
