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
			for (s32 z = 0; z < 3; z++)
			{
				for (s32 i = 0; i < 30 * 30; i++)
				{
					const s32 h = i / 30 - 15;
					const s32 w = i % 30 - 15;
					
					Transform transform = Transform::translation(Vec3(h, w, -z));
					transform.setScaling(0.2f);

					char* primitiveName = "Sphere";
					char* materialName = "Metal";
					if (RandomGenerator::uniform_real() < 0.5)
					{
						materialName = "Lambert";
					} 
					
					std::string objectName = "SphereObject"; objectName += std::to_string(i) += std::to_string(z);
					world.addObject(objectName.c_str(), primitiveName, materialName, transform);
				}
			}
			printf("Object Num in World : %d\n", world.getObjectNum());
		}

		//カメラのセット
		{
			Vec3 lookAt(0, 0, 0);
			Vec3 lookFrom(0,0,3.0f);
			Camera camera(lookFrom, lookAt, Vec3::unitY(), 20, f32(ResolutionW) / f32(ResolutionH), 0.0, (lookFrom - lookAt).length());
			world.setCamera(camera);
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
	RayTracingEngine::render(world, renderTarget, 1, 5);


	//------------------------------------------
	// 画像に出力して結果の確認
	//------------------------------------------
	renderTarget.saveRenderResult("./picture/result.ppm");

}