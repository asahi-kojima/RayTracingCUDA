#include <algorithm>
#include "engine.h"


RayTracingEngine::RayTracingEngine()
{
	cudaMalloc(&mCamera, sizeof(Camera));
}

RayTracingEngine::~RayTracingEngine()
{
	cudaDeviceSynchronize();
	cudaFree(mCamera);
}










__global__ void getCenters(Hittable** world, size_t objectNum, vec3* centerList)
{
	for (u32 i = 0; i < objectNum; i++)
	{
		centerList[i] = world[i]->calcAABB().getCenterPos();
	}
}


void sort_along_axis(std::vector<std::pair<vec3, u32>>& pairs, const u32 start, u32 end, u32 depth = 0)
{
	if (end - start == 1)
	{
		return;
	}

	std::sort(pairs.begin() + start, pairs.begin() + end, [depth](std::pair<vec3, u32>& pair0, std::pair<vec3, u32> pair1) {const u32 axis_of_sort = depth % 3; return pair0.first[axis_of_sort] < pair1.first[axis_of_sort]; });

	sort_along_axis(pairs, start, start + (end - start) / 2, depth + 1);
	sort_along_axis(pairs, start + (end - start) / 2, end, depth + 1);
}


void sortObjects(Hittable** world, size_t objectNum, u32* indexList)
{
	printf("Sort of World object start\n");
	
	//collect center info of objects;
	vec3 *centerList;
	CHECK(cudaMallocManaged(&centerList, sizeof(vec3) * objectNum));

	getCenters<<<1,1>>>(world, objectNum, centerList);
	GPU_ERROR_CHECKER(cudaPeekAtLastError());
	CHECK(cudaDeviceSynchronize());

	//sort
	std::vector<std::pair<vec3, u32> > pairs;
	for (u32 i = 0; i < objectNum; i++)
	{
		pairs.push_back({centerList[i], i});
	}

	//std::sort(pairs.begin(), pairs.end());
	sort_along_axis(pairs,0, objectNum);

	for (u32 i = 0; i < objectNum; i++)
	{
		indexList[i] = pairs[i].second;
	}


	cudaFree(centerList);
	printf("Sort of World object finish\n");
}

__global__ void make_node(Node* node, Hittable** world, size_t objectNum, u32* newOrderedIndexList)
{
	new (node) Node(world, newOrderedIndexList, 0, objectNum);
}

void RayTracingEngine::setObjects(const std::vector<Hittable*>& world)
{

	if (mRootNode)
	{
		cudaFree(mRootNode);
	}

	Hittable** hittableList;

	CHECK(cudaMallocManaged(&hittableList, sizeof(Hittable*) * world.size()));
	for (u32 i = 0, end = world.size(); i < end; i++)
	{
		hittableList[i] = world[i];
	}

	u32 *newOrderedIndexList;
	CHECK(cudaMallocManaged(&newOrderedIndexList, sizeof(u32) * world.size()));

	sortObjects(hittableList, world.size(),newOrderedIndexList);

	CHECK(cudaMalloc(&mRootNode, sizeof(Node)));
	make_node << <1, 1 >> > (mRootNode, hittableList, world.size(), newOrderedIndexList);
	CHECK(cudaDeviceSynchronize());
	GPU_ERROR_CHECKER(cudaPeekAtLastError());



	CHECK(cudaFree(newOrderedIndexList));
}

void RayTracingEngine::setCamera(const Camera& camera)
{
	cudaMemcpy(mCamera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
}

void RayTracingEngine::setRenderTarget(RenderTarget& target)
{
	mRenderTarget = target;
}




__device__ Color castRayAndCalcColor(Node* worldNode, const Ray& ray, const u32 maxDepth, SecondaryInfoByRay& secondaryInfoByRay)
{
	Color resultColor(0xFFFFFF);
	Ray current_ray = ray;

	u32 hitCounter = 0;
	
	for (u32 depth = 0; depth < maxDepth; depth++)
	{
		HitRecord rec;
		if (worldNode->hit(current_ray, 0.001f, MAXFLOAT, rec))
		{
			hitCounter++;

			Ray scattered;
			Color attenuation(0x000000);
			if (rec.material->scatter(current_ray, rec, attenuation, scattered))
			{
				resultColor *= attenuation;
				current_ray = scattered;
			}
			else
			{
				secondaryInfoByRay.depth = depth;
				return attenuation;
			}
		}
		else
		{
			vec3 direction = current_ray.direction();
			f32 length2 = direction.lengthSquared();
			f32 direction_y = direction[1];
			
			f32 t = 0.5f * (tanh(direction_y * 3) + 1.0f);
			//t = tanh(t < 0 ? 0 : (t > 1 ? 1 : t));
			//printf("%f : ", t);
			resultColor *= (Color(0xFFFFFF) * t + Color(0x444444) * (1.0f - t));

			secondaryInfoByRay.depth = depth;

			return (hitCounter == 0 ? Color(0xAAAAAA) : resultColor);
		}
	}

	secondaryInfoByRay.depth = maxDepth;
	return Color(0x000000);
}


__global__ void castRayToWorld(Node* worldNode, Color* pixels, Camera* camera, const u32 screenSizeW, const u32 screenSizeH, const u32 sampleSize, const u32 maxDepth)
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

		SecondaryInfoByRay additinalRayInfo;
		resultColor += castRayAndCalcColor(worldNode,ray, maxDepth, additinalRayInfo);
	}
	resultColor /= sampleSize;
	
	resultColor.clamp();

	*(pixels + pixelIndex) = resultColor;
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
}


void RayTracingEngine::saveRenderResult(const std::string& path)
{
	mRenderTarget.saveRenderResult(path);
}