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
			// for (s32 z = -3; z < 10; z++)
			// {
			// 	const s32 num = 30;
			// 	for (s32 i = 0; i < num * num; i++)
			// 	{
			// 		const s32 h = i / num - num/2;
			// 		const s32 w = i % num - num/2;

			// 		if (h == 0 && w == 0 && -z > 0)
			// 			continue;
			// 		f32 s = z * (M_PI) / 6;
			// 		f32 newH = cos(s) * h - sin(s) * w;
			// 		f32 newW = sin(s) * h + cos(s) * w;

			// 		Transform transform = Transform::translation(Vec3(newH, newW, -z));
			// 		transform.setRotationAngle(Vec3::generateRandomUnitVector() * 10);
			// 		transform.setScaling(0.2f);

			// 		char* primitiveName = "AABB";
			// 		char* materialName = "Metal";
			// 		if (RandomGenerator::uniform_real() < 0.3)
			// 		{
			// 			materialName = "Water";
			// 		} 

			// 		std::string objectName = "SphereObject"; objectName += std::to_string(i) += std::to_string(z);

			// 		SurfaceProperty property{};
			// 		property.setAlbedo(Color(RandomGenerator::uniform_int(0, 0xFFFFFF)));
			// 		world.addObject(objectName.c_str(), primitiveName, materialName, transform,property);
			// 	}
			// }
			
			constexpr f32 BoardScale = 555.0f;
			{
				Transform transform;
				transform.setScaling(BoardScale, 1, BoardScale);
				transform.setTranslation(BoardScale / 2, BoardScale, BoardScale / 2);

				SurfaceProperty property{};
				property.setAlbedo(Color::White);

				world.addObject("Ceil", "Board", "Lambert", transform, property);
			}

			{
				Transform transform;
				transform.setScaling(BoardScale, 1, BoardScale);
				transform.setTranslation(BoardScale / 2, 0, BoardScale / 2);

				SurfaceProperty property{};
				property.setAlbedo(Color::White);

				world.addObject("Floor", "Board", "Lambert", transform, property);
			}
			
			{
				Transform transform;
				transform.setScaling(BoardScale, 1, BoardScale);
				transform.setRotationAngle(M_PI_2, 0, 0);
				transform.setTranslation(BoardScale / 2, BoardScale, BoardScale / 2);

				SurfaceProperty property{};
				property.setAlbedo(Color::White);
				world.addObject("BackBoard", "Board", "Lambert", transform, property);
			}

			{
				Transform transform;
				transform.setScaling(BoardScale, 1, BoardScale);
				transform.setRotationAngle(0, 0, M_PI_2);
				transform.setTranslation(0, BoardScale / 2, BoardScale / 2);

				SurfaceProperty property{};
				property.setAlbedo(Color::Red);
				world.addObject("RightBoard", "Board", "Lambert", transform, property);
			}

			{
				Transform transform;
				transform.setScaling(BoardScale, 1, BoardScale);
				transform.setRotationAngle(0, 0, M_PI_2);
				transform.setTranslation(BoardScale, BoardScale / 2, BoardScale / 2);

				SurfaceProperty property{};
				property.setAlbedo(Color::Green);
				world.addObject("LeftBoard", "Board", "Lambert", transform, property);
			}


			printf("Object Num in World : %d\n", world.getObjectNum());
		}

		//カメラのセット
		{
			Vec3 lookFrom(278, 278, -800);
			Vec3 lookAt(278,278,0);
			Camera camera(lookFrom, lookAt, Vec3::unitY(), 40, f32(ResolutionW) / f32(ResolutionH), 0.0, 1);
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