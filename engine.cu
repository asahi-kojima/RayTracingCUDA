#include "engine.h"


__device__ bool getColorFromRay(Node* worldNode,Ray& current_ray, const u32 depth, const u32 maxDepth, Color& color)
{
	HitRecord rec;
	if (worldNode->hit(current_ray, 0.01, MAXFLOAT, rec))
	{
		Ray scattered;
		Color attenuation;
		if (depth < maxDepth && rec.material->scatter(current_ray, rec, attenuation, scattered))
		{
			color = attenuation;
			current_ray = scattered;
			return false;
		}
		else
		{
			color = Color(0x000000);
			return true;
		}
	}
	else
	{
		vec3 unitDirection = normalize(current_ray.direction());

		f32 t = 0.5f * (unitDirection.getY() + 1.0f);
		color = Color(0xFFFFFF) * (1.0f - t) + Color(0xF0FFFF) * t;
		return true;
	}
}


__device__ Color castRayAndCalcColor(Node* worldNode, const Ray& ray, const u32 maxDepth, SecondaryInfoByRay& secondaryInfoByRay)
{
	Color resultColor(0xFFFFFF);

	Ray currentRay = ray;
	u32 depth = 0;
	for (; depth < maxDepth; depth++)
	{
		Color colorFromThisRay;
		bool isRayTerminated = getColorFromRay(worldNode, currentRay, depth,maxDepth, colorFromThisRay);
		resultColor *= colorFromThisRay;

		if (isRayTerminated)
		{
			break;
		}
	}

	secondaryInfoByRay.depth = depth;
	return resultColor;
}


__global__ void castRayToWorld(Node* worldNode, Color* pixels, CameraCore* camera, const u32 screenSizeW, const u32 screenSizeH, const u32 sampleSize, const u32 maxDepth)
{
	const u32 id_w = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 id_h = blockIdx.y * blockDim.y + threadIdx.y;

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
		const f32 u = static_cast<f32>(id_w + RandomGeneratorGPU::signed_uniform_real() * 0.1f) * inv_screenSizeW;
		const f32 v = static_cast<f32>(id_h + RandomGeneratorGPU::signed_uniform_real() * 0.1f) * inv_screenSizeH;
		// const f32 u = static_cast<f32>(id_w) * inv_screenSizeW;
		// const f32 v = static_cast<f32>(id_h) * inv_screenSizeH;
		Ray ray = camera->getRay(u, v);

		SecondaryInfoByRay additinalRayInfo;
		resultColor += castRayAndCalcColor(worldNode,ray, maxDepth, additinalRayInfo);
	}
	resultColor /= sampleSize;
	

	*(pixels + pixelIndex) = resultColor;
}



RayTracingEngine::RayTracingEngine()
{
}

RayTracingEngine::~RayTracingEngine()
{
	cudaDeviceSynchronize();
}

__global__ void make_node(Node* node, Hittable** world, size_t objectNum)
{
	new (node) Node(world, objectNum);
}

void RayTracingEngine::setObjects(const std::vector<Hittable*>& world)
{

	if (mRootNode)
	{
		cudaFree(mRootNode);
	}

	Hittable** hittableList;
	CHECK(cudaMallocManaged(&hittableList, sizeof(Hittable*) * sizeof(world.size())));
	for (u32 i = 0, end = world.size(); i < end; i++)
	{
		hittableList[i] = world[i];
	}


	CHECK(cudaMalloc(&mRootNode, sizeof(Node)));

	make_node << <1, 1 >> > (mRootNode, hittableList, world.size());
	CHECK(cudaDeviceSynchronize());
	GPU_ERROR_CHECKER(cudaPeekAtLastError());
}

void RayTracingEngine::setCamera(const Camera& camera)
{
	mCamera = camera.pCore;
}

void RayTracingEngine::setRenderTarget(RenderTarget& target)
{
	mRenderTarget = target;
}

void RayTracingEngine::render(const u32 sampleSize, const u32 depth)
{
	printf("Rendering Start!\n");


	dim3 block(16, 16);
	dim3 grid((mRenderTarget.getResolutionWidth() + block.x - 1) / block.x, (mRenderTarget.getResolutionHeight() + block.y - 1) / block.y);
	castRayToWorld << <grid, block >> > (
		mRootNode,
		mRenderTarget.getPixels(), 
		mCamera, 
		mRenderTarget.getResolutionWidth(), 
		mRenderTarget.getResolutionHeight(),
		sampleSize,
		depth);

	CHECK(cudaDeviceSynchronize());
	//castRayToWorld << <grid, block >> > (mCamera);
}


void RayTracingEngine::saveRenderResult(const std::string& path)
{
	mRenderTarget.saveRenderResult(path);
}