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
	std::vector<Hittable *> world;

	const f32 world_scale = 20.0f;
	const f32 world_scale_x = world_scale;
	const f32 world_scale_y = world_scale;
	const f32 world_scale_z = world_scale;
	//まずは箱を作る
	{
		const f32 extention_x = world_scale_x / 2;
		const f32 extention_y = world_scale_y / 2;
		const f32 extention_z = world_scale_z / 2;
		const vec3 vertex_list[8] = {
		vec3(+extention_x, +extention_y, +extention_z) - vec3(0, extention_y, 0),
		vec3(-extention_x, +extention_y, +extention_z) - vec3(0, extention_y, 0),
		vec3(+extention_x, -extention_y, +extention_z) - vec3(0, extention_y, 0),
		vec3(-extention_x, -extention_y, +extention_z) - vec3(0, extention_y, 0),
		vec3(+extention_x, +extention_y, -extention_z) - vec3(0, extention_y, 0),
		vec3(-extention_x, +extention_y, -extention_z) - vec3(0, extention_y, 0),
		vec3(+extention_x, -extention_y, -extention_z) - vec3(0, extention_y, 0),
		vec3(-extention_x, -extention_y, -extention_z) - vec3(0, extention_y, 0)};

		const size_t index_list[5 * 2 * 3] = {
			1,0,3,
			0,2,3,
			0,4,2,
			4,6,2,
			5,1,7,
			1,3,7,
			4,5,6,
			5,7,6,
			3,2,7,
			2,6,7};
		
		for (s32 i = 0; i < 10; i++)
		{
			const s32 offset = 3 * i;
			const vec3& v0 = vertex_list[index_list[offset + 0]];
			const vec3& v1 = vertex_list[index_list[offset + 1]];
			const vec3& v2 = vertex_list[index_list[offset + 2]];
			world.push_back(make_object<Triangle>(v0, v1, v2, make_material<Lambertian>(make_texture<CheckerTexture>(Color(0xCCCCCC), Color::White, 0.5)), false));
		}
	}
	//水面を作る
	{
		const s32 PolygonNum = 70;
		const f32 diff_x = world_scale_x / PolygonNum;
		const f32 diff_z = world_scale_z / PolygonNum;

		vec3 displacement[4] = 
		{
			vec3(0, 0, 0),
			vec3(0, 0, +diff_z),
			vec3(+diff_x, 0, 0),
			vec3(+diff_x, 0, +diff_z)
		};

		for (s32 ix = 0; ix < PolygonNum; ix++)
		{
			for (s32 iz = 0; iz < PolygonNum; iz++)
			{
				const f32 offset_x = ix * diff_x - world_scale_x / 2;
				const f32 offset_z = iz * diff_z - world_scale_z / 2;
				const vec3 origin(offset_x, 0, offset_z);
				
				auto setHeightField = [](vec3& v) -> void 
				{
					const f32 x = v[0];
					const f32 z = v[2];
					v[1] = sin((x *cos(z)+ 5 * z) * 2 * M_PI) * 0.1;
				};
				vec3 v[4] = 
				{
					origin + displacement[0],origin + displacement[1],origin + displacement[2],origin + displacement[3]
				};
				for (u32 jx = 0; jx < 2; jx++)
				{
					for (u32 jz = 0; jz < 2; jz++)
					{
						if ((ix == 0 && jx == 0) || (ix == PolygonNum - 1 && jx == 1) || (iz == 0 && jz == 0) || (iz == PolygonNum - 1 && jz == 1))
							continue;
						setHeightField(v[2 * jx + jz]);
					}
				}
				Color color = Color::Azure;
				Material* material = make_material<Dielectric>(1.3, color);
				// Material* material = make_material<Dielectric>(1.3, Color(RandomGenerator::uniform_int(0, 0xFFFFFF)));
				world.push_back(make_object<Triangle>(v[0], v[3], v[2], material));
				world.push_back(make_object<Triangle>(v[0], v[1], v[3], material));
			}
		}
	}

	//オブジェクト
	{
		const s32 objectNumPerEdge = 10;
		const f32 objectRange = world_scale * 0.9;
		const f32 distanceBetweenObjects = objectRange / objectNumPerEdge;
		const f32 objectExtentionScale = distanceBetweenObjects * 0.2;
		const vec3 startingPoint(-objectRange / 2, -1, -objectRange / 2);
		for (u32 i = 0; i < objectNumPerEdge; i++)
		{
			for (u32 j = 0; j < objectNumPerEdge; j++)
			{
				const vec3 center = startingPoint + vec3(distanceBetweenObjects * i, 0, distanceBetweenObjects * j);
				const vec3 extention(objectExtentionScale, objectExtentionScale, objectExtentionScale);

				Material* material = make_material<Metal>(make_texture<ConstantTexture>(Color::Bronze));
				if ((i + j) % 2 == 0)
				{
					material = make_material<Metal>(make_texture<ConstantTexture>(Color::Blue));
				}
				world.push_back(make_object<AABB>(center - extention, center + extention, material));

			}
		}
	}

	//照明
	{
		vec3 center(0,100,0);
		vec3 extention = vec3(1, 1, 1) * 10000;
		world.push_back(make_object<AABB>(center - extention, center + extention,make_material<SunLight>(Color::White, 1.0)));
	}


	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 1;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(0,0,0);
	// vec3 lookFrom(0.5,0.2, 1);
	// vec3 lookFrom(0.9, 0.4, 1);
	vec3 lookFrom(1.0, 5, 3.0);
	lookFrom *= (20 / lookFrom.length());
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
	engine.render(30, 50);

	std::string s = "./build/result";
	s += std::to_string(0);
	s += ".ppm";
	engine.saveRenderResult(s);
}
