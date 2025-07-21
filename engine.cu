#include <chrono>
#include "engine.h"
#include "bvh_node.h"

__device__ Color castRayAndCalcColor(BvhNode* worldNode, const Ray& ray, const u32 maxDepth)
{
	Color resultColor(0xFFFFFF);
	Ray currentRay = ray;
	
	for (u32 depth = 0; depth < maxDepth; depth++)
	{
		HitRecord record;
		if (worldNode->isHit(currentRay, 0.001f, MAXFLOAT, record))
		{
			// レコードには本当に衝突したオブジェクトの情報が一部入っているので、
			// その情報を基にレコードを正確に更新する。
			{

				//法線の設定
				const Vec4 normal(record.normal, 0.0f);
				const Mat4& invTransposeTransformMat = record.hitObject->getTransform().getInvTransposeTransformMatrix();
				record.normal = (invTransposeTransformMat * normal).extractXYZ().normalize();
			}
			
            Ray scattered;
			Color albedo(0x000000);
			if (record.material->scatter(currentRay, record, albedo, scattered))
			{
				if (resultColor.isNan())
				{
					resultColor.printColor();
				}
				resultColor *= albedo;
				currentRay = scattered;
			}
			else
			{
				return albedo * resultColor;
			}
		}
		else
		{
			Vec3 direction = currentRay.direction();
			const f32 dx = direction[0];
			const f32 dy = direction[1];
			const f32 dz = direction[2];
			const Color backgroundColor = Color(abs(sin(5 * (dx + 4 * dz))), abs(cos(9 * dy * dz)), abs(cos(dy * 5 + sin(dy * 3)) * sin(dx + dz * cos(dy * dy))));
			resultColor *= backgroundColor;

			return resultColor;
		}
	}

	return Color(0x000000);
}

__global__ void castRayToWorld(BvhNode* worldNode, Color* pixels, Camera* camera, const u32 screenSizeW, const u32 screenSizeH, const u32 sampleSize, const u32 maxDepth)
{
	const u32 id_w = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 id_h = blockIdx.y * blockDim.y + threadIdx.y;
	if (id_h % 100 == 0 && id_w % 100 == 0) printf("%d , %d\n", id_w, id_h);
	if (id_w >= screenSizeW || id_h >= screenSizeH)
	{
		return;
	}

	const u32 pixelIndex = id_h * screenSizeW + id_w;
	
		const f32 inv_screenSizeW = 1.0f / static_cast<f32>(screenSizeW - 1);
		const f32 inv_screenSizeH = 1.0f / static_cast<f32>(screenSizeH - 1);
	
		Color resultColor = Color(0x000000);
		for (u32 s = 0; s < sampleSize; s++)
		{
			const f32 samplingRange = 0.01f;
		const f32 u = static_cast<f32>(id_w + RandomGeneratorGPU::signed_uniform_real() * samplingRange) * inv_screenSizeW;
		const f32 v = static_cast<f32>(id_h + RandomGeneratorGPU::signed_uniform_real() * samplingRange) * inv_screenSizeH;
			
			Ray ray = camera->getRay(u, v);
	
			resultColor += castRayAndCalcColor(worldNode,ray, maxDepth);
		}
		resultColor /= sampleSize;
		
	
		*(pixels + pixelIndex) = resultColor;
}

void RayTracingEngine::render(World& world, RenderTarget& renderTarget, const u32 sampleSize, const u32 depth)
{
	std::chrono::system_clock::time_point start, end;
	start = std::chrono::system_clock::now();
    printf("Rendering Start\n");



	dim3 block(16, 16);
	dim3 grid((renderTarget.getResolutionWidth() + block.x - 1) / block.x, (renderTarget.getResolutionHeight() + block.y - 1) / block.y);
	castRayToWorld << <grid, block >> > (
		world.getRootBvhDevicePtr(),
		renderTarget.getPixels(), 
		world.getCameraManagedPtr(), 
		renderTarget.getResolutionWidth(), 
		renderTarget.getResolutionHeight(),
		sampleSize,
		depth);

    KERNEL_ERROR_CHECKER;

    printf("Rendering Finish\n");
	end = std::chrono::system_clock::now();
	f32 time = static_cast<f32>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
	printf("Rendering Time = %fs\n", time);
}