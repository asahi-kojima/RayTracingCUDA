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
			for (s32 z = -1; z < 10; z++)
			{
				const s32 num = 15;
				for (s32 i = 0; i < num * num; i++)
				{
					
					const Vec3 angle = Vec3::generateRandomUnitVector() * 2 * M_PI;
					Transform transform = Transform::translation(Vec3(RandomGenerator::signed_uniform_real() * 4, RandomGenerator::signed_uniform_real() * 4, -z));
					f32 baseScale = 0.2f + RandomGenerator::uniform_real() * 0.1f;
					f32 scale = 0.8;
					for (s32 zz = 0; zz < 1; zz++)
					{
						//Transform transform = Transform::translation(Vec3::zero());
						transform.setRotationAngle(angle);
						transform.setScaling(baseScale *= scale);

						char* primitiveName = "AABB";
						char* materialName = "Diamond";
						if (RandomGenerator::uniform_real() < 0.5)
						{
							materialName = "Metal";
						}
						std::string objectName = "SphereObject"; objectName += std::string("-") += std::to_string(z) += std::string("-") += std::to_string(zz)+= std::string("-") += std::to_string(i);

						SurfaceProperty property{};
						property.setAlbedo(Color(RandomGenerator::uniform_int(0, 0xFFFFFF)));
						world.addObject(objectName.c_str(), primitiveName, materialName, transform,property);
						
					}
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
	for (u32 i = 0; i < 1; i++)
	RayTracingEngine::render(world, renderTarget, SampleNum, MaxDepth);

	//------------------------------------------
	// 画像に出力して結果の確認
	//------------------------------------------
	renderTarget.saveRenderResult("./picture/result.ppm");

}