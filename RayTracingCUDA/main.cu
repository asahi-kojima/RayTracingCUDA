#include <ctime>
#include "common.h"
#include "vector.h"
#include "mesh.h"
#include "geometry_generator.h"
#include "vertex.h"
#include "matrix.h"
#include "material.h"
#include "transform.h"
#include "scene.h"
#include "util.h"

//TODO: move to util.h
#include <curand_kernel.h>
constexpr u32 RANDOM_GENERATOR_STATE_COUNT = 32;
__device__ curandState s[32];

__global__ void setup_gpu(time_t time)
{
	const u32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < RANDOM_GENERATOR_STATE_COUNT)
	{
		curand_init(static_cast<unsigned long long>(time) + idx, 0, 0, &s[idx]);
	}
}


Transform generateRandomTransform(const f32 scale = 100.0f)
{
	Transform transform;
	transform.setTranslation(Vec3(RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real()) * scale);
	transform.setRotation(RandomGenerator::uniform_real(0, 5), Vec3::generateRandomUnitVector());
	transform.setScaling(1);
	return transform;
}


int main()
{
	setup_gpu << <1, RANDOM_GENERATOR_STATE_COUNT >> > (time(0));
	KERNEL_ERROR_CHECKER;

	Mesh sphereMesh      = GeometryGenerator::sphereGenerator(5, 5);
	Mesh tetrahedronMesh = GeometryGenerator::tetrahedronGenerator();
	Mesh octahedronMesh  = GeometryGenerator::octahedronGenerator();
	Mesh boxMesh         = GeometryGenerator::boxGenerator();
	Mesh geoSphereMesh0  = GeometryGenerator::geoSphereGenerator(0);
	Mesh geoSphereMesh1  = GeometryGenerator::geoSphereGenerator(1);
	Mesh geoSphereMesh2  = GeometryGenerator::geoSphereGenerator(2);
	Mesh geoSphereMesh3  = GeometryGenerator::geoSphereGenerator(3);
	Mesh geoSphereMesh4  = GeometryGenerator::geoSphereGenerator(4);
	Mesh planeMesh       = GeometryGenerator::planeGenerator(10);
	Mesh coneMesh        = GeometryGenerator::coneGenerator(20);
	Mesh cylinderMesh    = GeometryGenerator::cylinderGenerator(6);
	Mesh torusMesh       = GeometryGenerator::torusGenerator(0.05f, 100, 100);

	Material pureMetal{Material::MaterialType::METAL, 0.0f, 1.0, 1.0f, 0.0f};
	Material fuzzyMetal{Material::MaterialType::METAL, 0.2f, 0.0, 0.0f, 0.0f};
	Material water{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.1f, 0.0f};
	Material glass{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.5f, 0.0f};
	Material diamond{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 2.5f, 0.0f};
	Material pureLambertian{Material::MaterialType::LAMBERTIAN, 1.0f, 0.0, 0.0f, 0.0f};
	Material light{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 1, true};
	Material lowIntesityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 0.1, true};
	Material highIntensityLight{ Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 10, true };
	Material invisibleLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure, true, true};

	Scene scene;
	{
		scene.addMaterial("metal", pureMetal);
		scene.addMaterial("fuzzyMetal", fuzzyMetal);
		scene.addMaterial("water", water);
		scene.addMaterial("glass", glass);
		scene.addMaterial("diamond", diamond);
		scene.addMaterial("diffuse", pureLambertian);
		scene.addMaterial("light", light);
		scene.addMaterial("lowIntesityLight", lowIntesityLight);
		scene.addMaterial("highIntensityLight", highIntensityLight);
		scene.addMaterial("invisibleLight", invisibleLight);


		scene.addMesh("plane", planeMesh);
		scene.addMesh("sphere", sphereMesh);
		scene.addMesh("tetrahedron", tetrahedronMesh);
		scene.addMesh("octahedron", octahedronMesh);
		scene.addMesh("box", boxMesh);
		scene.addMesh("geoSphere0", geoSphereMesh0);
		scene.addMesh("geoSphere1", geoSphereMesh1);
		scene.addMesh("geoSphere2", geoSphereMesh2);
		scene.addMesh("geoSphere3", geoSphereMesh3);
		scene.addMesh("geoSphere4", geoSphereMesh4);
		scene.addMesh("cone", coneMesh);
		scene.addMesh("cylinder", cylinderMesh);
		scene.addMesh("torus", torusMesh);
	}

	const char* meshNameList[] = {
		"box",
		"sphere",
		"tetrahedron",
		"octahedron",
		"geoSphere0",
		"geoSphere1",
		"geoSphere2",
		"geoSphere3",
		"geoSphere4",
		"plane",
		"cone",
		"cylinder",
		"torus"
	};

	const char* materialNameList[] = {
		"metal",
		"fuzzyMetal",
		"water",
		"glass",
		"diamond",
		"diffuse",
		"light",
		//"highIntensityLight"
	};

	Result result;

	Group cornellBox("CornellBox");
	{

		constexpr f32 scale = 0.06f;
		constexpr f32 range = 3.0f;
		for (int i = 0; i < 5000; i++)
		{
			const f32 r = RandomGenerator::uniform_real(0.8, 1) * range;
			const f32 phi = RandomGenerator::uniform_real(0, 2 * M_PI);
			const f32 theta = acosf(2 * RandomGenerator::uniform_real(0, 1) - 1);
			const f32 x = r * sin(theta) * cos(phi);
			const f32 y = r * sin(theta) * sin(phi);
			const f32 z = r * cos(theta);

			result = cornellBox.addChildObject(Object{
				std::string("box") + std::to_string(i),
				"box",
				RandomGenerator::uniform_real() < 0.5 ? "fuzzyMetal" : "metal",
				Transform(Vec3(x, y, z), Vec3(1, 1, 1) * scale, Quaternion(0, Vec3::unitZ())),
				SurfaceProperty{RandomGenerator::uniform_real() < 0.5 ? Color::Blue : Color::Bronze}});
		}



		//constexpr f32 LightScale = 1.0f;
		//constexpr f32 LightRange = 100;
		//for (int i = 0; i < 1000; i++)
		//{
		//	const f32 r = LightRange;
		//	const f32 phi = RandomGenerator::uniform_real(0, 2 * M_PI);
		//	const f32 theta = acosf(2 * RandomGenerator::uniform_real(0, 1) - 1);
		//	const f32 x = r * sin(theta) * cos(phi);
		//	const f32 y = r * sin(theta) * sin(phi);
		//	const f32 z = r * cos(theta);

		//	result = cornellBox.addChildObject(Object{
		//		std::string("light") + std::to_string(i),
		//		"geoSphere1",
		//		"invisibleLight",
		//		Transform(Vec3(x, y, z), Vec3(1, 1, 1) * LightScale, Quaternion(0, Vec3::unitZ())),
		//		SurfaceProperty{Color::random()} });
		//}

		//{
		//	constexpr f32 LightSizeScale = 0.3f;
		//	result = cornellBox.addChildObject(Object{
		//		"Light",
		//		"sphere",
		//		"diamond",
		//		Transform(Vec3::zero(), Vec3::one() * range * 0.95, Quaternion(0, Vec3::unitZ())),
		//		SurfaceProperty{Color::White} });
		//}

		{
			constexpr f32 LightSizeScale = 0.3f;
			result = cornellBox.addChildObject(Object{
				"Light",
				"torus",
				"invisibleLight",
				Transform(Vec3::zero(), Vec3::one() * 50, Quaternion(-0, Vec3::unitZ())),
				SurfaceProperty{Color::Bronze} });
		}
	}
	
	scene.addGroup(cornellBox);



	result = scene.build();
	result = scene.initLaunchParams();
	result = scene.render();
	cudaDeviceSynchronize();
}