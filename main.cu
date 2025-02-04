#include <stdio.h>
#include <vector>
#include <memory>
#include <vector>
#include <string>
#include "vector.h"
#include "color.h"
#include "ray.h"
#include "object.h"
#include "util.h"
#include "hittable.h"
#include "texture.h"
#include "engine.h"



int main()
{
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 30);
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



		// world.push_back(make_object<Sphere>(vec3(0, -3000, 0), 3000.0f, make_material<Lambertian>(make_object<CheckerTexture>(std::make_unique<ConstantTexture>(Color::Green), std::make_unique<ConstantTexture>(Color::Azure)))));
		world.push_back(make_object<Sphere>(vec3(0, -3000, 0), 3000.0f, make_material<Metal>(Color::Gray, 0.9f)));

		world.push_back(make_object<Sphere>(vec3(-12, 1, 2), 1.0f, make_material<QuasiGravitationalField>(10.0f, vec3(-12, 1, 2))));
		world.push_back(make_object<Sphere>(vec3(-8, 1, 0), 1.0f, make_material<Metal>(Color(1, 1, 0.2), 0)));
		world.push_back(make_object<Sphere>(vec3(-4, 1, 0), 1.0f, make_material<Dielectric>(1.5f)));
		world.push_back(make_object<Sphere>(vec3(-4, 1, 0), -0.9f, make_material<Dielectric>(1.5f)));
		world.push_back(make_object<Sphere>(vec3(0, 1, 0), 1.0f, make_material<Metal>(Color::Gold, 0)));
		world.push_back(make_object<Sphere>(vec3(4, 1, 1), 1.0f, make_material<Rutherford>(1.0f, vec3(4, 1, 1))));
		world.push_back(make_object<Sphere>(vec3(8, 1, 0), 1.0f, make_material<Metal>(Color::Bronze, 1)));
		world.push_back(make_object<Sphere>(vec3(0, 1, -2), 1.0f, make_material<Metal>(Color(0x000FA0), 0.3)));

#else
		constexpr s32 Range = 10;
		for (s32 w = -Range; w <= Range; w+= 1)
		{
			for (s32 h = -Range; h <= Range; h+=1)
			{
				for (s32 z = 0; z < 10; z++)
				{
					f32 which = RandomGenerator::uniform_real();
					vec3 pos(w, h, -z);

					Material* material;
					if (which < 0.95 && !(w == 0 && h == 0 && z == 0))
					{
						material = make_material<Metal>(RandomGenerator::uniform_int(0, 0xFFFFFF), 0.0f);
					}
					else
					{
						material = make_material<GravitationalField>(RandomGenerator::uniform_real(1.0f, 5.0f), pos);
					}
					world.push_back(make_object<Sphere>(pos, 0.25f, std::move(material)));

				}
			}
		}


#endif
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
	RenderTarget renderTarget(resolutionX, resolutionY);

	//=================================================================
	// オブジェクトの準備
	//=================================================================
	RayTracingEngine engine;

	engine.setObjects(world);
	engine.setCamera(camera);
	engine.setRenderTarget(renderTarget);
	engine.render();

	engine.saveRenderResult("./build/result.ppm");
}
