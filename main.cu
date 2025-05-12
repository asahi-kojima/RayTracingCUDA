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
	cudaDeviceSetLimit(cudaLimitStackSize, 1024*100);
	setup_gpu<<<1,1>>>();
	//=================================================================
	// オブジェクトの準備
	//=================================================================
	std::vector<Hittable*> world;
	{
		constexpr s32 Range = 10;
		for (s32 w = -Range; w <= Range; w+= 1)
		{
			for (s32 h = -Range; h <= Range; h+=1)
			{
				for (s32 z = -5; z <= 10; z++)
				{
					f32 which = RandomGenerator::uniform_real();
					const f32 scale = 1.0f;
					const vec3 pos = vec3(w , h  , -z) * scale;

					Material* material = nullptr;

					if (h == 0 && w == 0 && z == 0)
					{
						//material = make_material<GravitationalField>(1.2, pos);
						//world.push_back(make_object<AABB>(vec3(-0.1, -0.1, -0.1),vec3(0.1, 0.1, 0.1), material));
						continue;
					}
					else if (h == 0 && w == 0)
					{
						continue;
					}
					else if (h == 0 || w == 0)
					{
						continue;
					}
					// else if (h == 0 && w == 0)
					// {
					// 	material = make_material<Rutherford>(10.5, pos);
					// }
					else
					{
						// material = make_material<Metal>(RandomGenerator::uniform_int(0, 0xFFFFFF),0.8f);
						material = make_material<Metal>(RandomGenerator::uniform_int(0, 0xFFFFFF));
						vec3 diff_x(RandomGenerator::signed_uniform_real(),RandomGenerator::signed_uniform_real(),RandomGenerator::signed_uniform_real());
						vec3 diff_y(RandomGenerator::signed_uniform_real(),RandomGenerator::signed_uniform_real(),RandomGenerator::signed_uniform_real());
						f32 diff_scale = 0.1f;
						world.push_back(make_object<AABB>(pos + vec3(-0.1, -0.1, -0.1)*2 + diff_x * diff_scale,pos + vec3(0.1, 0.1, 0.1)*2 + diff_y * diff_scale, material));
						continue;
					}
					
					world.push_back(make_object<Sphere>(pos, 0.1f, material));
				}
			}
		}
		

	}
	// {
	// 	const vec3 pos = vec3(0 , 0  , -40);
	// 	Material* material = make_material<Dielectric>(1.2);
	// 	f32 scale = 300;
	// 	world.push_back(make_object<AABB>(pos + vec3(-1, -1, -0.1) * scale,pos + vec3(1, 1, 0.1) * scale, material));
	// 	//world.push_back(make_object<Sphere>(pos, 30, material));
	// }
	{
		const vec3 pos = vec3(0 , 0  , -115);
		Material* material = make_material<Metal>(Color(0xFFFFFF));
		f32 scale = 300;
		//world.push_back(make_object<AABB>(pos + vec3(-0.1, -0.1, -0.1) * scale,pos + vec3(0.1, 0.1, 0.1) * scale, material));
		world.push_back(make_object<Sphere>(pos, 100, material));
	}


	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 1;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(0, 0, 0);
	//vec3 lookFrom(13, 2, 5);
	vec3 lookFrom(0,0,2.0f);


	Camera camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());

	//=================================================================
	// レンダーターゲットの準備
	//=================================================================
	RenderTarget renderTarget[3] = {RenderTarget(resolutionX, resolutionY),RenderTarget(resolutionX, resolutionY),RenderTarget(resolutionX, resolutionY) };

	//=================================================================
	// オブジェクトの準備
	//=================================================================
	RayTracingEngine engine;

	engine.setObjects(world);
	engine.setRenderTarget(renderTarget[0]);

	camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.1, (lookFrom - lookAt).length());
	engine.setCamera(camera);
	engine.render(30, 50);

	std::string s = "./build/result";
	s += std::to_string(0);
	s += ".ppm";
	engine.saveRenderResult(s);
}
