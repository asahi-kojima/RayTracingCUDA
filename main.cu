#include <curand_kernel.h>
#include <stdio.h>
#include "world.h"
#include "render_target.h"
#include "engine.h"

//__device__ curandState s[32];


int main(int argc, char** argv)
{
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitStackSize, 1024*100);
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
		{
			for (u32 i = 0; i < 8000; i++)
			{
				Transform transform = Transform::translation(Vec3::generateRandomUnitVector() * 10);
				transform.setScaling(0.01f, 0.01f, 0.01f);
				//std::cout << "T = " << transform.getTransformMatrix()(0, 3) << std::endl;
				char* primitiveName = "Sphere";
				char* materialName = "Metal";
				if (RandomGenerator::uniform_real() < 0.5)
				{
					materialName = "Lambert";
				} 

				std::string objectName = "SphereObject"; objectName += std::to_string(i);
				world.addObject(objectName.c_str(), primitiveName, materialName, transform);
			}
			printf("Object Num in World : %d\n", world.getObjectNum());
		}

		//カメラのセット
		{
			Vec3 lookAt(0, 0, 0);
			Vec3 lookFrom(0,0,20.0f);
			Camera(lookFrom, lookAt, Vec3::unitY(), 20, f32(ResolutionW) / f32(ResolutionH), 0.0, (lookFrom - lookAt).length());
		}

		world.buildBvh();
	}



	//------------------------------------------
	// レンダーターゲットの準備
	//------------------------------------------
	RenderTarget renderTarget(ResolutionW, ResolutionH);


	//------------------------------------------
	// エンジンに渡して、レンダリング	
	//------------------------------------------
	RayTracingEngine::render(world, renderTarget, 5, 5);

	//------------------------------------------
	// 画像に出力して結果の確認
	//------------------------------------------
	renderTarget.saveRenderResult("./build/result.ppm");

}