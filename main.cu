#include <curand_kernel.h>
#include <stdio.h>
#include "world.h"
#include "render_target.h"
#include "engine.h"

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
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitStackSize, 1024*100);
	setup_gpu<<<1,1>>>();
	KERNEL_ERROR_CHECKER;
	//------------------------------------------
	// レンダリングする際の解像度を外から与える
	//------------------------------------------
	if (argc <= 4)
	{
		printf("few arguments\n");
		exit(1);
	}

	const u32 ResolutionW = atoi(argv[1]);
	const u32 ResolutionH = atoi(argv[2]);
	const u32 SampleNum = atoi(argv[3]);
	const u32 MaxDepth = atoi(argv[4]);


	//------------------------------------------
	// ワールドを準備
	//------------------------------------------
	World world{};
	{
		//オブジェクトの追加
		{
			for (s32 z = -3; z < 10; z++)
			{
				const s32 num = 30;
				for (s32 i = 0; i < num * num; i++)
				{
					const s32 h = i / num - num/2;
					const s32 w = i % num - num/2;

					if (h == 0 && w == 0)
						continue;
					
					Transform transform = Transform::translation(Vec3(h, w, -z));
					transform.setRotationAngle(Vec3::generateRandomUnitVector() * 10);
					transform.setScaling(0.2f);

					char* primitiveName = "AABB";
					char* materialName = "Metal";
					if (RandomGenerator::uniform_real() < 0.3)
					{
						materialName = "Diamond";
					} 

					std::string objectName = "SphereObject"; objectName += std::to_string(i) += std::to_string(z);

					SurfaceProperty property{};
					property.setAlbedo(Color(RandomGenerator::uniform_int(0, 0xFFFFFF)));
					world.addObject(objectName.c_str(), primitiveName, materialName, transform,property);
				}
			}

			printf("Object Num in World : %d\n", world.getObjectNum());
		}

		//カメラのセット
		{
			Vec3 lookAt(0, 0, 0);
			Vec3 lookFrom(0,0,5.0f);
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
	for (u32 i = 0; i < 1; i++)
	RayTracingEngine::render(world, renderTarget, SampleNum, MaxDepth);

	//------------------------------------------
	// 画像に出力して結果の確認
	//------------------------------------------
	renderTarget.saveRenderResult("./picture/result.ppm");

}