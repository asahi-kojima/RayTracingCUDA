#include "engine.h"
#include "bvh_node.h"

__device__ Color castRayAndCalcColor(BvhNode* worldNode, const Ray& ray, const u32 maxDepth)
{
	Color resultColor(0xFFFFFF);
	Ray current_ray = ray;
	
	for (u32 depth = 0; depth < maxDepth; depth++)
	{
		HitRecord rec;
		if (worldNode->isHit(current_ray, 0.001f, MAXFLOAT, rec))
		{
            Ray scattered;
			Color attenuation;
			if (rec.material->scatter(current_ray, rec, attenuation, scattered))
			{
				resultColor *= attenuation;
				current_ray = scattered;
			}
			else
			{
				return Color(0x000000);
			}
		}
		else
		{
			Vec3 direction = current_ray.direction();
			f32 length2 = direction.lengthSquared();
			f32 direction_y = direction[1];
	
			f32 t = 0.5f * (direction_y * direction_y / length2 + 1.0f);
			resultColor *= Color(0xFFFFFF) * (1.0f - t) + Color(0xF0FFFF) * t;

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
}