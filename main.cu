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



__global__ void f(Texture* p)
{
	p = new ConstantTexture(Color(0, 0, 4));
	Color c = p->color(0, 0, vec3(0,0, 0));
	printf("%p\n", p);
	printf("%f, %f, %f\n", c.r(), c.g(), c.b());
}

__global__ void g(Ray* p)
{
	vec3 d = p->direction();
	printf("%f, %f, %f\n", d[0], d[1], d[2]);
}

__global__ void testprint(int i = 0)
{
	printf("asahi kojima : %d\n", i);
}


int main()
{
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024);

	//=================================================================
	// オブジェクトの準備
	//=================================================================
	Hittable* object0 = make_object<Sphere>(vec3(0, -3000, 0), 3000.0f, make_material<Metal>(0xFF00F0, 0.0f));
	Hittable* object1 = make_object<Sphere>(vec3(0, 2, 0), 2.0f, make_material<Metal>(0xFF00F0, 0.0f));
	Hittable* object2 = make_object<Sphere>(vec3(4, 1, 1), 1.0f, make_material<Metal>(0xFF00F0, 0.0f));
	std::vector<Hittable*> hittableObjectList;
	{
		hittableObjectList.push_back(make_object<Sphere>(vec3(0, -3000, 0), 3000.0f, make_material<Metal>(0xFF00F0, 0.0f)));
		hittableObjectList.push_back(make_object<Sphere>(vec3(0, 2, 0), 2.0f, make_material<Metal>(0xFF00F0, 0.0f)));
		hittableObjectList.push_back(make_object<Sphere>(vec3(4, 1, 1), 1.0f, make_material<Metal>(0x0F00F0, 0.0f)));
	}
	// Hittable** hittableList;
	// CHECK(cudaMallocManaged(&hittableList, sizeof(Hittable*) * sizeof(3)));
	// hittableList[0] = object0;
	// hittableList[1] = object1;
	// hittableList[2] = object2;

	//Hittable* hittableList[3] = {object0, object1, object2};
	//Hittable** gpuList;
	//CHECK(cudaMalloc(&gpuList, sizeof(hittableList)));
	//CHECK(cudaMemcpy(gpuList, hittableList, sizeof(hittableList), cudaMemcpyHostToDevice));

	//=================================================================
	// カメラの準備
	//=================================================================
	constexpr f32 BaseResolution = 1.0f * 2.0f / 2;
	const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
	const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

	vec3 lookAt(0, 0, 0);
	vec3 lookFrom(13, 2, 5);

	//Camera* camera = new Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());
	Camera camera = Camera(lookFrom, lookAt, vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());


	//=================================================================
	// レンダーターゲットの準備
	//=================================================================
	RenderTarget renderTarget(resolutionX, resolutionY);

	//=================================================================
	// オブジェクトの準備
	//=================================================================
	RayTracingEngine engine;

	engine.setObjects(hittableObjectList);
	engine.setCamera(camera);
	engine.setRenderTarget(renderTarget);
	engine.render();

	engine.saveRenderResult("./build/result.ppm");
}
