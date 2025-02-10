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

// /usr/local/cuda/bin/nvcc -std=c++20 -rdc=true -O3 -DNDEBUG -w *.cu && ./a.out && convert ./build/result.ppm ./build/rayTracingDemo.png

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
	cudaDeviceSetLimit(cudaLimitStackSize, 1024*4);
	setup_gpu<<<1,1>>>();
	//=================================================================
	// オブジェクトの準備
	//=================================================================
	std::vector<Hittable*> world;
	{
#if 0
		constexpr f32 Range = 25;
		constexpr u32 Dense = 25;
		constexpr f32 Interval = 2 * Range / Dense;
		constexpr f32 Radius = 0.1 * Interval;
		for (f32 x = -Range; x < Range; x += Interval)
		{
			for (f32 z = -Range; z < Range; z += Interval)
			{
				vec3 center(x + 0.3f * RandomGenerator::signed_uniform_real(), Radius, z + 0.3 * RandomGenerator::signed_uniform_real());

				world.push_back(make_object<Sphere>(
					center,
					Radius,
					make_material<Metal>(
						Color(
							RandomGenerator::uniform_real(),
							RandomGenerator::uniform_real(),
							RandomGenerator::uniform_real()),
						0)));

			}
		}




		world.push_back(make_object<Sphere>(vec3(0, -1000, 0), 1000.0f, make_material<Lambertian>(Color::Gray)));
		
		world.push_back(make_object<Sphere>(vec3(-12, 1, 2), 1.0f, make_material<QuasiGravitationalField>(10.0f, vec3(-12, 1, 2))));
		world.push_back(make_object<Sphere>(vec3(-8, 1, 0), 1.0f, make_material<Metal>(Color(1, 1, 0.2), 0)));
		world.push_back(make_object<Sphere>(vec3(-4, 1, 0), 1.0f, make_material<Rutherford>(10.0f, vec3(-4, 1, 0))));
		world.push_back(make_object<Sphere>(vec3(0, 1.0, 0), 1.0f, make_material<Dielectric>(1.1f)));
		world.push_back(make_object<Sphere>(vec3(0, 1.0, 0), -0.95f, make_material<Dielectric>(1.5f)));
		world.push_back(make_object<Sphere>(vec3(4, 1, 0), 1.0f, make_material<Metal>(Color::Gold, 0)));
		world.push_back(make_object<Sphere>(vec3(8, 1, 0), 1.0f, make_material<Dielectric>(2)));
		world.push_back(make_object<Sphere>(vec3(0, 1, -4), 1.0f, make_material<Metal>(Color::Bronze, 1)));
		


		// world.push_back(make_object<Sphere>(vec3(0, -1000, 0), 1000.0f, make_material<Lambertian>(Color(0.5f, 0.5f, 0.5f))));
		// for (s32 a = -11; a < 11; a++)
		// {
		// 	for (s32 b = -11; b < 11; b++)
		// 	{
		// 		f32 choose_mat = RandomGenerator::uniform_real();
		// 		vec3 center(a + 0.9 * RandomGenerator::uniform_real(), 0.2, b + 0.9 * RandomGenerator::uniform_real());
		// 		if ((center - vec3(4, 0.2, 0)).length() > 0.9)
		// 		{
		// 			Material* material;
		// 			if (choose_mat < 0.8)
		// 			{
		// 				material = make_material<Metal>(RandomGenerator::uniform_int(0, 0xFFFFFF),0.0f);
		// 			}
		// 			else
		// 			{
		// 				material = make_material<Dielectric>(1.5f);
		// 			}
		// 			world.push_back(make_object<Sphere>(center, 0.2f, material));
		// 		}
		// 	}
		// }

		// world.push_back(make_object<Sphere>(vec3(0, 1, 0), 1.0f, make_material<Dielectric>(1.5f)));
		// world.push_back(make_object<Sphere>(vec3(-4, 1, 0), 1.0f, make_material<Lambertian>(Color(0.4, 0.2, 0.1f))));
		// world.push_back(make_object<Sphere>(vec3(4, 1, 0), 1.0f, make_material<Metal>(Color(0.7f, 0.6f, 0.5f), 0)));

#else
		constexpr s32 Range = 10;
		for (s32 w = -Range; w <= Range; w+= 1)
		{
			for (s32 h = -Range; h <= Range; h+=1)
			{
				for (s32 z = -Range; z <= Range; z++)
				{
					f32 which = RandomGenerator::uniform_real();
					vec3 pos(w, h, -z);

					Material* material;
					// if (which < 0.95 && !(w == 0 && h == 0 && z == 0))
					// {
					// 	material = make_material<Metal>(RandomGenerator::uniform_int(0, 0xFFFFFF),1.0f);
					// }
					// else
					// {
					// 	material = make_material<Rutherford>(3.5f, pos);
					// }
					if ((w + h + z) % 2 != 0)
					{
						material = make_material<Metal>(RandomGenerator::uniform_int(0, 0xFFFFFF),0.0f);
					}
					else
					{
						material = make_material<Rutherford>(3.5f, pos);
					}
					world.push_back(make_object<Sphere>(pos, 0.125f, material));

				}
			}
		}


#endif
	}


	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 2;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(0, 0, 0);
	vec3 lookFrom(13, 2, 5);
	//vec3 lookFrom(0,0,2.0f);


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

	for (u32 i = 0, maxI = 10; i < maxI; i++)
	{
		printf("%d : times\n", i);
		f32 phi = i *  (2 * M_PI) / maxI;
		vec3 lookAt(0, 0, 0);
		vec3 lookFrom(0.5f * sin(phi), 0, 0.5f * cos(phi));
		//vec3 lookFrom(14 * sin(phi), 2, 14 * cos(phi));
		//vec3 lookFrom(13, 2 + i, 5);
		camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());
		engine.setCamera(camera);
		engine.render(30, 50);

		std::string s = "./build/result";
		s += std::to_string(i);
		s += ".ppm";
		engine.saveRenderResult(s);
	}
}
