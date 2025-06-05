#include <stdio.h>
#include <vector>
#include <memory>
#include <vector>
#include <string>
#include <curand_kernel.h>
#include <iostream>
#include "world.h"

//__device__ curandState s[32];


int main(int argc, char** argv)
{
	//------------------------------------------
	// レンダリングする際の解像度を外から与える
	//------------------------------------------
	if (argc <= 2)
	{
		printf("few arguments\n");
		exit(1);
	}

	const u32 ResolutionW = atoi(argv[1]);
	const u32 ResolutionH = atoi(argv[2]);


	//------------------------------------------
	// ワールドを準備
	//------------------------------------------
	World world{};
	{
		//オブジェクトの追加
		for (u32 i = 0; i < 8000; i++)
		{
			Transform transform = Transform::translation(Vec3::generateRandomUnitVector() * 10);
			char* primitiveName = "Sphere";
			char* materialName = "Metal";
			if (RandomGenerator::uniform_real() < 0.5)
			{
				materialName = "Lambert";
			} 

			std::string objectName = "SphereObject"; objectName += std::to_string(i);
			world.addObject(objectName.c_str(), primitiveName, materialName, transform);
		}
		std::cout << "Object Num in World : " << world.getObjectNum() << std::endl;
		
		world.buildBvh();
	}



	//------------------------------------------
	// レンダーターゲットの準備
	//------------------------------------------



	//------------------------------------------
	// エンジンに渡して、レンダリング	
	//------------------------------------------


	//------------------------------------------
	// 画像に出力して結果の確認
	//------------------------------------------

}



// /usr/local/cuda/bin/nvcc --generate-code arch=compute_86,code=sm_86 -std=c++17 -rdc=true -O3 -DNDEBUG -w *.cu && ./a.out && convert ./build/result.ppm ./build/rayTracingDemo.png



// __global__ void setup_gpu()
// {
// 	for (u32 i = 0; i < 32; i++)
// 	{
// 		curand_init(static_cast<unsigned long long>(i), 0, 0, &s[i]);
// 	}
// }

// int main(int argc, char** argv)
// {
// 	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
// 	cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 100);
// 	setup_gpu<<<1, 1>>>();



// 	if (argc <= 2)
// 	{
// 		printf("Screen Resolution not specified!\n");
// 		printf("This program will be terminated\n");
// 		return 1;
// 	}
// 	const u32 resolutionW = std::stoi(argv[1]);
// 	const u32 resolutionH = std::stoi(argv[2]);

// 	//=================================================================
// 	// オブジェクトの準備
// 	//=================================================================
// 	World world;

// 	Primitive sphereMesh;
// 	world.addPrimitive("Sphere", sphereMesh);
// 	Primitive boxMesh;
// 	world.addPrimitive("box", boxMesh);
	
// 	{
// 		Transform transform;	
// 		Object object_sphere("Sphere", transform);
// 		world.addObject(object_sphere);
// 	}
// 	{
// 		Transform transform;
// 		Object object_sphere("Sphere", transform);
// 		world.addObject(object_sphere);
// 	}
	
// 	{
// 		Transform transform;	
// 		Object object_sphere("box", transform);
// 		world.addObject(object_sphere);
// 	}


// 	//=================================================================
// 	// カメラの準備
// 	//=================================================================
// 	// constexpr f32 BaseResolution = 1.0f * 2.0f / 1;
// 	// const u32 resolutionX = static_cast<u32>(1920 * BaseResolution);
// 	// const u32 resolutionY = static_cast<u32>(1080 * BaseResolution);

// 	Vec3 lookAt(0,0,0);
// 	Vec3 lookFrom(1.0, 5, 3.0);
// 	Camera camera = Camera(lookFrom, lookAt, Vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, (lookFrom - lookAt).length());

// 	//=================================================================
// 	// レンダーターゲットの準備
// 	//=================================================================
// 	RenderTarget renderTarget[3] = {RenderTarget(resolutionX, resolutionY), RenderTarget(resolutionX, resolutionY), RenderTarget(resolutionX, resolutionY)};

// 	//=================================================================
// 	// オブジェクトの準備
// 	//=================================================================
// 	RayTracingEngine engine;

// 	engine.setObjects(world);
// 	engine.setRenderTarget(renderTarget[0]);

// 	camera = Camera(lookFrom, lookAt, Vec3(0, 1, 0), 20, f32(resolutionX) / f32(resolutionY), 0.0, 2 * (lookFrom - lookAt).length());
// 	engine.setCamera(camera);
// 	engine.render(30, 50);

// 	std::string s = "./build/result";
// 	s += std::to_string(0);
// 	s += ".ppm";
// 	engine.saveRenderResult(s);
// }
