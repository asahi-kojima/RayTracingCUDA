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
	Mesh planeMesh       = GeometryGenerator::planeGenerator(1);
	Mesh coneMesh        = GeometryGenerator::coneGenerator(20);
	Mesh cylinderMesh    = GeometryGenerator::cylinderGenerator(10);
	Mesh torusMesh       = GeometryGenerator::torusGenerator(0.05f, 40, 20);

	Material pureMetal{Material::MaterialType::METAL, 0.0f, 1.0, 1.0f, 0.0f};
	Material fuzzyMetal{Material::MaterialType::METAL, 0.2f, 0.0, 0.0f, 0.0f};
	Material water{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.1f, 0.0f};
	Material glass{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.5f, 0.0f};
	Material diamond{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 2.5f, 0.0f};
	Material pureLambertian{Material::MaterialType::LAMBERTIAN, 1.0f, 0.0, 0.0f, 0.0f};
	Material light{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 1, true};
	Material lowIntesityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 0.1, true};
	Material highIntesityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 10, true};

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
		scene.addMaterial("highIntensityLight", highIntesityLight);


		scene.addMesh("sphere", sphereMesh);
		scene.addMesh("tetrahedron", tetrahedronMesh);
		scene.addMesh("octahedron", octahedronMesh);
		scene.addMesh("box", boxMesh);
		scene.addMesh("geoSphere0", geoSphereMesh0);
		scene.addMesh("geoSphere1", geoSphereMesh1);
		scene.addMesh("geoSphere2", geoSphereMesh2);
		scene.addMesh("geoSphere3", geoSphereMesh3);
		scene.addMesh("geoSphere4", geoSphereMesh4);
		scene.addMesh("plane", planeMesh);
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
		"highIntensityLight"
	};

	Result result;



	scene.addObject(Object{ "torus", "geoSphere2", "diffuse", Transform(Vec3(-3, 1, 0), 1), SurfaceProperty{Color::White} });
	scene.addObject(Object{ "torus", "geoSphere0", "diffuse", Transform(Vec3(0, 1, -3), 1, Quaternion(0, Vec3(0,1,0))), SurfaceProperty{Color::White}});
	scene.addObject(Object{ "torus", "sphere", "diffuse", Transform(Vec3(0, 1, 3), 1, Quaternion(0, Vec3(0,1,0))), SurfaceProperty{Color::White} });
	scene.addObject(Object{ "torus", "torus", "diamond", Transform(Vec3(0, 1, 0), 1, Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())), SurfaceProperty{Color::Red} });
	//scene.addObject(Object{ "torus", "geoSphere2", "glass", Transform(Vec3(3, 1, 0), -0.9), SurfaceProperty{Color::White} });
	scene.addObject(Object{ "torus", "octahedron", "metal", Transform(Vec3(3, 1.2, 0), 1, Quaternion(RandomGenerator::uniform_real(0, 10),Vec3::generateRandomUnitVector())), SurfaceProperty{Color::Bronze} });

	scene.addObject(Object{ "torus", "sphere", "glass", Transform(Vec3(0, 1, 0), 0.6), SurfaceProperty{Color::White} });
	
	


	Transform trans;
	trans.setScaling(1000);
	trans.setTranslation(Vec3(0, -1000, 0));
	scene.addObject(Object{ "object1", "box", "diffuse", trans, SurfaceProperty{Color::Gray}});

	const f32 skyScale = 10000;
	trans.setScaling(skyScale, skyScale, skyScale);
	trans.setTranslation(Vec3(0, 0, 0));
	scene.addObject(Object{ "object1", "box", "lowIntesityLight", trans });

	for (s32 a = -100; a < 11; a++)
	{
		for (s32 b = -100; b < 11; b++)
		{
			auto A = 1 * a;
			auto B = 1 * b;
			Vec3 center{ A + 0.9f * RandomGenerator::uniform_real(), 0.3f, B + 0.9f * RandomGenerator::uniform_real() };
			

			Object object
			{
				std::string("obj") + std::to_string(a) + std::to_string(b),
				meshNameList[RandomGenerator::uniform_int(0, sizeof(meshNameList) / sizeof(meshNameList[0]))],
				 materialNameList[RandomGenerator::uniform_int(0, sizeof(materialNameList) / sizeof(materialNameList[0]))],
				 Transform(center, 0.3, Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())) ,
				 SurfaceProperty{ Color::random()}
			};
			scene.addObject(object);
		}
	}

	//Group group0("group0");
	//{
	//	for (s32 k = -1; k <= 1; k++)
	//	{
	//		for (s32 i = -1; i <= 1; i++)
	//		{
	//			for (s32 j = -1; j <= 1; j++)
	//			{
	//				trans.setTranslation(Vec3(2 * i, 2 * k, 2 * j));
	//				trans.setScaling(Vec3(1,1,1) * 0.8);
	//				Object object
	//				{
	//					std::string("group0") + std::to_string(i) + std::string("-") + std::to_string(j) + std::string("-") + std::to_string(k),
	//					std::string("box"),
	//					"metal",
	//					trans,
	//					SurfaceProperty{ Color::Bronze }
	//				};

	//				group0.addChildObject(object);
	//			}
	//		}
	//	}
	//}

	//trans.setScaling(Vec3(1,1,1) * 0.3);
	//trans.setTranslation(Vec3(0, 1, 0));
	//trans.setRotation(Vec3::generateRandomUnitVector(), 10);
	//group0.setTransform(trans);
	//scene.addGroup(group0, Transform::translation(Vec3(0, -3, 0)));
	
	//for (u32 i = 0; i < 100; i++)
	//{
	//	result = scene.addObject(object0,generateRandomTransform(), object0.getName() + std::to_string(i));
	//	result = scene.addObject(object1,generateRandomTransform(), object1.getName() + std::to_string(i));
	//	result = scene.addObject(object2,generateRandomTransform(), object2.getName() + std::to_string(i));
	//	result = scene.addObject(object3, generateRandomTransform(), object3.getName() + std::to_string(i));

	//	result = scene.addGroup(group0, generateRandomTransform(), object0.getName() + std::to_string(i));
	//	result = scene.addGroup(group1, generateRandomTransform(), group1.getName() + std::to_string(i));
	//	result = scene.addGroup(group2, generateRandomTransform(), group2.getName() + std::to_string(i));
	//	result = scene.addGroup(group3, generateRandomTransform(), group3.getName() + std::to_string(i));
	//	result = scene.addGroup(group3, generateRandomTransform(), group3.getName() + std::to_string(i));
	//	result = scene.addGroup(group3, generateRandomTransform(), group3.getName() + std::to_string(i));
	//	result = scene.addGroup(group0, generateRandomTransform(), object0.getName() + std::to_string(i));
	//}


	result = scene.build();
	result = scene.initLaunchParams();
	result = scene.render();
	cudaDeviceSynchronize();
}