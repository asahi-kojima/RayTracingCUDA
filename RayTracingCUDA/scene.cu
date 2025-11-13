#include <iostream>
#include "scene.h"
#include "util.h"
#include <Windowsnumerics.h>


__constant__ GpuRayTracingLaunchParams gGpuRayTracingLaunchParams = {};




Result Scene::initLaunchParams()
{
	std::cout << "===================================================" << std::endl;
	std::cout << "Data Copy & GpuRayTracingLaunchParams Setting Start" << std::endl;
	std::cout << "===================================================" << std::endl;

	//GpuRayTracingLaunchParams gpuRayTracingLaunchParamsHostSide;

	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.vertexArray,       sizeof(float3)             * mRayTracingDataOnCPU.vertexArray.size()));
	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.triangleIndexArray,        sizeof(uint3)              * mRayTracingDataOnCPU.triangleIndexArray.size()));
	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.normalArray,       sizeof(float3)              * mRayTracingDataOnCPU.normalArray.size()));
	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.materialArray,     sizeof(Material)           * mRayTracingDataOnCPU.materialArray.size()));
	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.instanceDataArray, sizeof(DeviceInstanceData) * mRayTracingDataOnCPU.instanceDataArray.size()));
	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.blasArray,         sizeof(BVHNode)            * mRayTracingDataOnCPU.blasArray.size()));
	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.tlasArray,         sizeof(BVHNode)            * mRayTracingDataOnCPU.tlasArray.size()));

	printf("vertex   array malloc : %8d byte\n", sizeof(float3)             * mRayTracingDataOnCPU.vertexArray.size());
	printf("index    array malloc : %8d byte\n", sizeof(uint3)              * mRayTracingDataOnCPU.triangleIndexArray.size());
	printf("normal   array malloc : %8d byte\n", sizeof(float3)             * mRayTracingDataOnCPU.normalArray.size());
	printf("material array malloc : %8d byte\n", sizeof(Material)           * mRayTracingDataOnCPU.materialArray.size());
	printf("instance array malloc : %8d byte\n", sizeof(DeviceInstanceData) * mRayTracingDataOnCPU.instanceDataArray.size());
	printf("blas     array malloc : %8d byte\n", sizeof(BVHNode)            * mRayTracingDataOnCPU.blasArray.size());
	printf("tlas     array malloc : %8d byte\n", sizeof(BVHNode)            * mRayTracingDataOnCPU.tlasArray.size());


	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.vertexArray,       mRayTracingDataOnCPU.vertexArray.data(),       sizeof(float3)             * mRayTracingDataOnCPU.vertexArray.size(),       cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.triangleIndexArray,        mRayTracingDataOnCPU.triangleIndexArray.data(),        sizeof(uint3)              * mRayTracingDataOnCPU.triangleIndexArray.size(),        cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.normalArray,       mRayTracingDataOnCPU.normalArray.data(),       sizeof(float3)             * mRayTracingDataOnCPU.normalArray.size(),       cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.materialArray,     mRayTracingDataOnCPU.materialArray.data(),     sizeof(Material)           * mRayTracingDataOnCPU.materialArray.size(),     cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.instanceDataArray, mRayTracingDataOnCPU.instanceDataArray.data(), sizeof(DeviceInstanceData) * mRayTracingDataOnCPU.instanceDataArray.size(), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.blasArray,         mRayTracingDataOnCPU.blasArray.data(),         sizeof(BVHNode)            * mRayTracingDataOnCPU.blasArray.size(),         cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(mGpuRayTracingLaunchParamsHostSide.tlasArray,         mRayTracingDataOnCPU.tlasArray.data(),         sizeof(BVHNode)            * mRayTracingDataOnCPU.tlasArray.size(),         cudaMemcpyHostToDevice));
	

	mGpuRayTracingLaunchParamsHostSide.vertexCount   = mRayTracingDataOnCPU.vertexArray.size();
	mGpuRayTracingLaunchParamsHostSide.indexCount    = mRayTracingDataOnCPU.triangleIndexArray.size();
	mGpuRayTracingLaunchParamsHostSide.normalCount   = mRayTracingDataOnCPU.normalArray.size();
	mGpuRayTracingLaunchParamsHostSide.materialCount = mRayTracingDataOnCPU.materialArray.size();
	mGpuRayTracingLaunchParamsHostSide.instanceCount = mRayTracingDataOnCPU.instanceDataArray.size();
	mGpuRayTracingLaunchParamsHostSide.blasCount     = mRayTracingDataOnCPU.blasArray.size();
	mGpuRayTracingLaunchParamsHostSide.tlasCount     = mRayTracingDataOnCPU.tlasArray.size();


	mGpuRayTracingLaunchParamsHostSide.pixelSizeVertical = 500;
	mGpuRayTracingLaunchParamsHostSide.pixelSizeHorizontal = 500;
	mGpuRayTracingLaunchParamsHostSide.invPixelSizeVertical = 1.0f / static_cast<f32>(mGpuRayTracingLaunchParamsHostSide.pixelSizeVertical);
	mGpuRayTracingLaunchParamsHostSide.invPixelSizeHorizontal = 1.0f / static_cast<f32>(mGpuRayTracingLaunchParamsHostSide.pixelSizeHorizontal);


	CHECK(cudaMalloc(&mGpuRayTracingLaunchParamsHostSide.renderTargetImageArray, sizeof(Color) * mGpuRayTracingLaunchParamsHostSide.pixelSizeVertical * mGpuRayTracingLaunchParamsHostSide.pixelSizeHorizontal));


	mGpuRayTracingLaunchParamsHostSide.frameCount = 0;

	Camera camera{Vec3(10, 10, 10), Vec3::zero(), Vec3::unitY(), 20, 1};
	mGpuRayTracingLaunchParamsHostSide.camera = camera;

	cudaMemcpyToSymbol(gGpuRayTracingLaunchParams, &mGpuRayTracingLaunchParamsHostSide, sizeof(GpuRayTracingLaunchParams));

	KERNEL_ERROR_CHECKER;


	return Result();
}










struct HitRecord
{
	bool isHit = false;
	f32 t;
	float3 hitPoint;
	float3 hitPointNormal;
	u32 objectID;
	u32 triangleID;

	__device__ HitRecord(bool isHit = false)
		: isHit(isHit)
	{
	}
	__device__ operator bool() const { return isHit; }
};


namespace
{
	struct TriangleIntersectionResult
	{
		bool isIntersected;
		f32 t;
		f32 alpha;
		f32 beta;

		__device__ TriangleIntersectionResult(bool isIntersected = false, f32 t = 0.0f, f32 alpha = 0.0f, f32 beta = 0.0f)
			: isIntersected(isIntersected)
			, t(t)
			, alpha(alpha)
			, beta(beta)
		{
		}
		__device__ operator bool() const { return isIntersected; }
	};
}

__device__ TriangleIntersectionResult intersectionTriangle(const Ray& ray, const float3& v0, const float3& v1, const float3& v2)
{
	const Vec3 p1 = Vec3(v1) - Vec3(v0);
	const Vec3 p2 = Vec3(v2) - Vec3(v0);
	const Vec3 v0ToO = ray.origin() - Vec3(v0);

	const Vec3 a0 = -ray.direction();
	const Vec3 a1 = p1;
	const Vec3 a2 = p2;

	const Vec3 cross1x2 = Vec3::cross(a1, a2);
	const Vec3 cross2x0 = Vec3::cross(a2, a0);
	const Vec3 cross0x1 = Vec3::cross(a0, a1);

	const f32 det = Vec3::dot(cross1x2, a0);
	if (isEqualF32(det, 0.0f))
	{
		return TriangleIntersectionResult{false};
	}
	
	
	const f32 t = Vec3::dot(cross1x2, v0ToO) / det;
	const f32 alpha = Vec3::dot(cross2x0, v0ToO) / det;
	const f32 beta = Vec3::dot(cross0x1, v0ToO) / det;

	const f32 tmin = ray.tmin();
	const f32 tmax = ray.tmax();

	if (!(t > tmin && t < tmax && alpha + beta < 1 && alpha > 0 && beta > 0))
	{
		return TriangleIntersectionResult{ false };
	}

	bool isCulling = false;//TODO

	return TriangleIntersectionResult{ true, t, alpha, beta };
}


__device__ s32 traceBlasTree(Ray& ray, const u32 blasRootIndex, const u32 vertexOffset, const u32 indexOffset)
{
	AABB::AABBIntersectionResult aabbHitResult;

	s32 closestTriangleID = -1;

	u32 nodeStack[32];
	u32 stackTop = 0;

	nodeStack[stackTop++] = blasRootIndex;
	while (stackTop > 0)
	{
		const u32 currentNodeIndex = nodeStack[--stackTop];
		BVHNode& currentNode = gGpuRayTracingLaunchParams.blasArray[currentNodeIndex];

		if (!(aabbHitResult = currentNode.aabb.doIntersect(ray)))
		{
			// 衝突がなければ以降の深度でも当たらないからスキップ
			continue;
		}

		//--------------------------------------------------------------
		// 衝突したのでrayのtmaxを更新する
		//--------------------------------------------------------------
		ray.tmax() = aabbHitResult.tmax;

		if (currentNode.primitiveCount > 0)
		{
			// BLASのリーフノードに到達 = メッシュの中の数個の三角形まで到達
			// 三角形との衝突判定を行う
			// ここでカレントノードが持っているfirstPrimitiveOffsetとprimitiveCountは一個のメッシュ内の三角形に対してのもの（全メッシュの配列内のインデックスではない
			// つまりtriangleIndexはローカルな三角形インデックス
			for (u32 triangleIndex = currentNode.firstPrimitiveOffset, end = currentNode.firstPrimitiveOffset + currentNode.primitiveCount; triangleIndex < end; triangleIndex++)
			{
				const uint3& index = gGpuRayTracingLaunchParams.triangleIndexArray[triangleIndex + indexOffset];//offsetがいる
				const float3& v0   = gGpuRayTracingLaunchParams.vertexArray[index.x + vertexOffset];
				const float3& v1   = gGpuRayTracingLaunchParams.vertexArray[index.y + vertexOffset];
				const float3& v2   = gGpuRayTracingLaunchParams.vertexArray[index.z + vertexOffset];

				if (TriangleIntersectionResult triangleIntersectionResult{}; triangleIntersectionResult = intersectionTriangle(ray, v0, v1, v2))
				{
					//ray.tmax()に衝突点の情報が入っている
					ray.tmax() = triangleIntersectionResult.t;
					closestTriangleID = triangleIndex;
				}
			}
		}
		else
		{
			//TODO : 深度が深くなりすぎた場合
			if (stackTop + 2 >= 32)
			{
				printf("Stack Overflow in BLAS traversal\n");
				break;
			}
			nodeStack[stackTop++] = currentNode.leftChildOffset;
			nodeStack[stackTop++] = currentNode.rightChildOffset;
		}
	}

	return closestTriangleID;
}


__device__ HitRecord traceTlasTree(Ray ray)
{
	HitRecord hitRecord;
	s32 closestInstanceID = -1;
	s32 closestTriangleID = -1;

	AABB::AABBIntersectionResult aabbHitResult;

	u32 nodeStack[32];
	u32 stackTop = 0;

	nodeStack[stackTop++] = 0;
	while (stackTop > 0)
	{
		//printf("Top = %d\n", stackTop);
		const u32 currentNodeIndex = nodeStack[--stackTop];
		BVHNode& currentNode = gGpuRayTracingLaunchParams.tlasArray[currentNodeIndex];
		

		//--------------------------------------------------------------
		// まずはこのTLASノードのAABBとの衝突を確認する
		//--------------------------------------------------------------
		if (!(aabbHitResult = currentNode.aabb.doIntersect(ray)))
		{
			// 衝突がなければ以降の深度でも当たらないからスキップ
			continue;
		}

		//--------------------------------------------------------------
		// 衝突したのでrayのtmaxを更新する
		//--------------------------------------------------------------
		ray.tmax() = aabbHitResult.tmax;

		//--------------------------------------------------------------
		// 末端であれば葉のインスタンス達を調べ、そうでなければ子に行く
		//--------------------------------------------------------------
		if (currentNode.primitiveCount > 0)
		{
			//TODO;
			for (u32 instanceID = currentNode.firstPrimitiveOffset, end = currentNode.firstPrimitiveOffset + currentNode.primitiveCount; instanceID < end; instanceID++)
			{
				const DeviceInstanceData& currentInstanceData = gGpuRayTracingLaunchParams.instanceDataArray[instanceID];

				if (aabbHitResult = currentInstanceData.aabb.doIntersect(ray))
				{
					//--------------------------------------------------------------------------------------
					// 衝突したのでrayのtmaxを更新する
					//--------------------------------------------------------------------------------------
					ray.tmax() = aabbHitResult.tmax;

					//--------------------------------------------------------------------------------------
					// インスタンスのAABBに衝突したので、これからインスタンスが参照するBLASツリーを探索する
					//--------------------------------------------------------------------------------------
					const u32 blasRootIndex = currentInstanceData.blasRootNodeIndex;
					
					//--------------------------------------------------------------------------------------
					// メッシュのローカル空間にレイを変換する
					//--------------------------------------------------------------------------------------
					Ray localRay = ray.transformWith(currentInstanceData.invTransformMat);
					
					//--------------------------------------------------------------------------------------
					// BLASツリーの探索
					//--------------------------------------------------------------------------------------
					const s32 tmpClosestTriangleID = traceBlasTree(localRay, blasRootIndex, currentInstanceData.vertexOffset, currentInstanceData.indexOffset);
					
					if (tmpClosestTriangleID >= 0)
					{
						// 三角形に衝突したのでパラメータ上限を更新
						ray.tmax() = localRay.tmax();

						closestInstanceID = instanceID;
						closestTriangleID = tmpClosestTriangleID;
					}
				}
			}
		}
		else
		{
			//TODO : 深度が深くなりすぎた場合
			if (stackTop + 2 >= 32)
			{
				printf("Stack Overflow in TLAS traversal\n");
				break;
			}
			nodeStack[stackTop++] = currentNode.leftChildOffset;
			nodeStack[stackTop++] = currentNode.rightChildOffset;
		}
	}

	//--------------------------------------------------------------
	// インスタンスIDが正なら衝突があったという事
	//--------------------------------------------------------------
	if (closestInstanceID >= 0)
	{
		/*
		struct HitRecord
		{
			f32 t;
			float3 hitPoint;
			float3 hitPointNormal;
			u32 materialID;
		};
		*/
		hitRecord.isHit = true;
		hitRecord.t = ray.tmax();
		hitRecord.hitPoint = ray.pointAt(hitRecord.t).toFloat3();
		
		hitRecord.hitPointNormal = gGpuRayTracingLaunchParams.normalArray[closestTriangleID];
		{
			const Mat4& transform = gGpuRayTracingLaunchParams.instanceDataArray[closestInstanceID].transformMat;
			
			hitRecord.hitPoint = (transform * Vec4(hitRecord.hitPoint, 1)).extractXYZ().toFloat3();
			
			const Mat4& normalTransform = gGpuRayTracingLaunchParams.instanceDataArray[closestInstanceID].normalTransformMat;

			hitRecord.hitPointNormal = (normalTransform * Vec4(hitRecord.hitPointNormal, 0)).extractXYZ().normalize().toFloat3();
			
			const Vec3& direction = ray.direction();
			const Vec3 normal = Vec3(hitRecord.hitPointNormal);
			if (Vec3::dot(direction, normal) > 0)
			{
				hitRecord.hitPointNormal = (normal * -1).toFloat3();
			}
		}
		hitRecord.objectID = closestInstanceID;
		hitRecord.triangleID = closestTriangleID;
	}

	return hitRecord;
}

__device__ float3 operator+(const float3& v, const float3& w)//delete
{
	return float3{ v.x + w.x, v.y + w.y, v.z + w.z };
}

__device__ __host__ float3 operator*(const float3& v, const f32 s)//delete
{
	return float3{ v.x * s, v.y * s, v.z * s };
}

__device__ __host__ float3 operator*(const float3& v, const float3& w)//delete
{
	return float3{ v.x * w.x, v.y * w.y, v.z * w.z };
}


__device__ __host__ float3& operator+=(float3& v, const float3& w)//delete
{
	v.x += w.x;
	v.y += w.y;
	v.z += w.z;
	return v;
}

__device__ bool shader(const Ray& ray, const HitRecord& hitRecord, const Material& material, Ray& scatteredRay, float3& albedo)
{
	// Diffuse shader
	const f32 diffuse = material.diffuse;

	const float3 target = hitRecord.hitPoint + hitRecord.hitPointNormal + Vec3::generateRandomUnitVector().toFloat3();
	scatteredRay = Ray(Vec3(hitRecord.hitPoint), Vec3::normalize(Vec3(target) - Vec3(hitRecord.hitPoint)));


	//Metal shader
	const Vec3 reflected = Vec3::reflect(ray.direction().normalize(), Vec3::normalize(hitRecord.hitPointNormal));
	scatteredRay = Ray(Vec3(hitRecord.hitPoint), reflected * diffuse + Vec3::normalize(Vec3(target) - Vec3(hitRecord.hitPoint) * (1 - diffuse)));

	albedo = float3{ material.albedo.r(), material.albedo.g(), material.albedo.b() };


	return true;
}

__device__ Color tracePath(Ray ray)
{
	float3 pathRadiance = float3{ 0.0f, 0.0f, 0.0f };
	float3 pathAttenuation = float3{1.0f, 1.0f, 1.0f};
	HitRecord hitRecord;
	
	const u32 maxBounce = 5;
	for (u32 bounce = 0; bounce < maxBounce; bounce++)
	{

		if (!(hitRecord = traceTlasTree(ray)))
		{
			float3 backGroundColor = float3{ 0.5f, 0.7f, 1.0f };
			pathRadiance += (backGroundColor * pathAttenuation);
			break;
		}
		Material material = gGpuRayTracingLaunchParams.materialArray[gGpuRayTracingLaunchParams.instanceDataArray[hitRecord.objectID].materialID];

		Ray scatteredRay;
		float3 albedo;
		if (!shader(ray, hitRecord, material, scatteredRay, albedo))
		{
			break;
		}

		pathAttenuation = pathAttenuation * albedo;
		ray = scatteredRay;

		const f32 maxIntensity = fmaxf(fmaxf(pathAttenuation.x, pathAttenuation.y), pathAttenuation.z);
		if (maxIntensity < 0.01f)
		{
			break;
		}
	}


	return Color(pathRadiance.x, pathRadiance.y, pathRadiance.z);
}



__global__ void raytracingKernel()
{
	const u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
	const u32 pixelID = yid * gGpuRayTracingLaunchParams.pixelSizeHorizontal + xid;

	if (xid >= gGpuRayTracingLaunchParams.pixelSizeHorizontal || yid >= gGpuRayTracingLaunchParams.pixelSizeVertical)
	{
		return;
	}



	const f32 samplingRange = 0.01f;
	const f32 u = static_cast<f32>(xid + RandomGeneratorGPU::signed_uniform_real() * samplingRange) * gGpuRayTracingLaunchParams.invPixelSizeHorizontal;
	const f32 v = static_cast<f32>(yid + RandomGeneratorGPU::signed_uniform_real() * samplingRange) * gGpuRayTracingLaunchParams.invPixelSizeVertical;
	Ray ray = gGpuRayTracingLaunchParams.camera.getRay(u, v);

	Color color = tracePath(ray);
	
	if (gGpuRayTracingLaunchParams.frameCount == 0)
	{
		gGpuRayTracingLaunchParams.renderTargetImageArray[pixelID] = color;
	}
	else
	{
		gGpuRayTracingLaunchParams.renderTargetImageArray[pixelID] = color;
	}
}





#include <fstream>
Result Scene::render()
{
	std::cout << "===================================================" << std::endl;
	std::cout << "                  Rendering Start                  " << std::endl;
	std::cout << "===================================================" << std::endl;
	//TLASの探索

	//Objectのローカル空間に移行し、BLASを探索する


    Result result;

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 block(16, 16);
    dim3 grid(
    (mGpuRayTracingLaunchParamsHostSide.pixelSizeHorizontal + block.x - 1) / block.x,
    (mGpuRayTracingLaunchParamsHostSide.pixelSizeVertical + block.y - 1) / block.y);

    raytracingKernel <<<grid, block >>> ();
    KERNEL_ERROR_CHECKER;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Rendering Time: " << elapsedTime << " ms : " << static_cast<s32>(1000.0f / elapsedTime) << " fps" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);





	const u32 ScreenWidth = mGpuRayTracingLaunchParamsHostSide.pixelSizeHorizontal;
	const u32 ScreenHeight = mGpuRayTracingLaunchParamsHostSide.pixelSizeVertical;

	Color* renderTarget;
	renderTarget = new Color[ScreenWidth * ScreenHeight];

	cudaMemcpy(renderTarget, mGpuRayTracingLaunchParamsHostSide.renderTargetImageArray, sizeof(Color) * ScreenWidth * ScreenHeight, cudaMemcpyDeviceToHost);

	std::ofstream outputFile("renderResult.ppm");
	outputFile << "P3\n" << ScreenWidth << " " << ScreenHeight << "\n255\n";
	for (s32 yid = ScreenHeight - 1; yid >= 0; yid--)
	{
		for (u32 xid = 0; xid < ScreenWidth; xid++)
		{
			const u32 index = yid * ScreenWidth + xid;
			Color& col = renderTarget[index];
			col = Color(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			outputFile << static_cast<s32>(255.99 * col[0]) << " " << static_cast<s32>(255.99 * col[1]) << " " << static_cast<s32>(255.99 * col[2]) << "\n";
		}
	}
	outputFile.close();

    return result;
}