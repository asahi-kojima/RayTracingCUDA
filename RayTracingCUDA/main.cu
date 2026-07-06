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

	Mesh sphereMesh = GeometryGenerator::sphereGenerator(50, 50);
	Mesh tetrahedronMesh = GeometryGenerator::tetrahedronGenerator();
	Mesh octahedronMesh = GeometryGenerator::octahedronGenerator();
	Mesh boxMesh = GeometryGenerator::boxGenerator();
	Mesh geoSphereMesh0 = GeometryGenerator::geoSphereGenerator(0);
	Mesh geoSphereMesh1 = GeometryGenerator::geoSphereGenerator(1);
	Mesh geoSphereMesh2 = GeometryGenerator::geoSphereGenerator(2);
	Mesh geoSphereMesh3 = GeometryGenerator::geoSphereGenerator(3);
	Mesh geoSphereMesh4 = GeometryGenerator::geoSphereGenerator(4);
	Mesh planeMesh = GeometryGenerator::planeGenerator(10);
	Mesh coneMesh = GeometryGenerator::coneGenerator(20);
	Mesh cylinderMesh = GeometryGenerator::cylinderGenerator(6);
	Mesh torusMesh = GeometryGenerator::torusGenerator(0.05f, 100, 100);

	Material pureMetal{ Material::MaterialType::METAL, 0.0f, 1.0, 1.0f, 0.0f };
	Material fuzzyMetal{ Material::MaterialType::METAL, 0.2f, 0.0, 0.0f, 0.0f };
	Material water{ Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.1f, 0.0f };
	Material glass{ Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.5f, 0.0f };
	Material diamond{ Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 2.5f, 0.0f };
	Material pureLambertian{ Material::MaterialType::LAMBERTIAN, 1.0f, 0.0, 0.0f, 0.0f };
	Material light{ Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 1, true };
	Material lowIntesityLight{ Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 0.1, true };
	Material highIntensityLight{ Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 10, true };
	Material invisibleLight{ Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 1, true, true };
	Material invisibleWeakLight{ Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 0.1f, true, true };

	Material emissiveMetal{ Material::MaterialType::METAL, 0.0f, 1.0, 1.0f, 0.0f, Color::White * 0.3, true, false };

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
		scene.addMaterial("invisibleWeakLight", invisibleWeakLight);
		scene.addMaterial("emissiveMetal", emissiveMetal);


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

	Group objects("CornellBox");
	{
		result = objects.addChildObject(Object{
			"floor",
			"plane",
			"metal",
			Transform(Vec3::zero(), Vec3(10000, 1, 10000)),
			SurfaceProperty{Color::Gray } });


		u32 index = 0;
		for (auto vertex : geoSphereMesh1.getVertexArray())
		{
			auto pos = vertex.position;
			pos = pos * 5;

			auto color = Color::Red;
			
			result = objects.addChildObject(Object{
				std::string("out") + std::to_string(index++),
				"box",
				"metal",
				Transform(pos, Vec3(1000, 0.01, 0.01)),//, Quaternion(RandomGenerator::uniform_real(0, 5), Vec3::generateRandomUnitVector())),
				SurfaceProperty{color} });


			result = objects.addChildObject(Object{
				std::string("out") + std::to_string(index++),
				"box",
				"metal",
				Transform(pos, Vec3(0.01, 0.01, 1000)),//, Quaternion(RandomGenerator::uniform_real(0, 5), Vec3::generateRandomUnitVector())),
				SurfaceProperty{color} });

			result = objects.addChildObject(Object{
				std::string("out") + std::to_string(index++),
				"box",
				"metal",
				Transform(pos, Vec3(0.01, 1000, 0.01)),//, Quaternion(RandomGenerator::uniform_real(0, 5), Vec3::generateRandomUnitVector())),
				SurfaceProperty{color} });
		}

	
		for (int i = 0; i < 10; i++)
		{
			
			


			//result = objects.addChildObject(Object{
			//	std::string("out") + std::to_string(i),
			//	"geoSphere3",
			//	"diamond",
			//	Transform(pos, scale * 1.1, Quaternion(RandomGenerator::uniform_real(0, 5), Vec3::generateRandomUnitVector())),
			//	SurfaceProperty{RandomGenerator::uniform_real() < 0.5f ? Color::White : Color::White} });

			Vec3 pos = Vec3(
				RandomGenerator::uniform_real(-1, 1) * 5,
				1,
				RandomGenerator::uniform_real(-1, 1) * 5
			);

			result = objects.addChildObject(Object{
				std::string("out") + std::to_string(i),
				"sphere",
				"metal",
				Transform(pos, Vec3::one() * 0.5),//, Quaternion(RandomGenerator::uniform_real(0, 5), Vec3::generateRandomUnitVector())),
				SurfaceProperty{RandomGenerator::uniform_real() < 0.5f ? Color::Blue : Color::Gold} });
		}

		{
			Vec3 pos = Vec3::zero();

			Vec3 scale = Vec3::one() * 100;

			result = objects.addChildObject(Object{
				"1",
				"sphere",
				"emissiveMetal",
				Transform(pos, scale, Quaternion(0, Vec3::unitZ())),
				SurfaceProperty{Color::White} });
		}


		{
			Vec3 pos = Vec3::zero();

			Vec3 scale = Vec3::one() * 400;

			result = objects.addChildObject(Object{
				"Light",
				"box",
				"invisibleLight",
				Transform(pos, scale, Quaternion(0, Vec3::unitZ())),
				SurfaceProperty{Color::White} });
		}
	}

	scene.addGroup(objects);



	result = scene.build();
	result = scene.initLaunchParams();
	result = scene.render("renderResult.ppm");
	cudaDeviceSynchronize();
}