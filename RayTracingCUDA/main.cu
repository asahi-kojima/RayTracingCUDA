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
	Mesh cylinderMesh    = GeometryGenerator::cylinderGenerator(6);
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
		//"highIntensityLight"
	};

	Result result;


	//for (s32 z = -1; z < 10; z++)
	//{
	//	const s32 num = 5;
	//	for (s32 i = 0; i < num * num; i++)
	//	{
	//		const s32 h = i / num - num / 2;
	//		const s32 w = i % num - num / 2;

	//		if (h == 0 && w == 0)
	//			continue;

	//		const f32 scale = 0.1f + RandomGenerator::uniform_real(-1, 1) * 0.05;

	//		std::string objectName = "SphereObject"; objectName += std::to_string(i) += std::string("-") += std::to_string(z);

	//		objectName += "_2";

	//		scene.addObject(Object{
	//			objectName,
	//			"box",
	//			"metal",
	//			Transform(Vec3(
	//			RandomGenerator::signed_uniform_real() * 2,
	//			RandomGenerator::signed_uniform_real() * 2, -z), scale, Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())),
	//			SurfaceProperty{Color::random()}
	//			});
	//	}
	//}

	//for (s32 i = 0; i < 400; i++)
	//{
	//	const f32 scale = 0.2f;

	//	std::string objectName = "GroundPlan2e" + std::to_string(i);
	//	scene.addObject(Object{
	//		objectName,
	//		"cylinder",
	//		"diamond",
	//		Transform(Vec3(RandomGenerator::signed_uniform_real() * 1, RandomGenerator::signed_uniform_real() * 1, RandomGenerator::signed_uniform_real() * 1)
	//			, Vec3(0.005, 10000, 0.005), Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())),
	//		SurfaceProperty{Color::White}
	//		});
	//}
	//for (s32 i = 0; i < 100; i++)
	//{
	//	const f32 scale = 0.2f;

	//	std::string objectName = "GroundPlan2e" + std::to_string(i);
	//	scene.addObject(Object{
	//		objectName,
	//		"torus",
	//		"metal",
	//		Transform(Vec3(RandomGenerator::signed_uniform_real() * 1, RandomGenerator::signed_uniform_real() * 1, RandomGenerator::signed_uniform_real() * 1)
	//			, Vec3(RandomGenerator::uniform_real(0.1, 0.2), 0.1, RandomGenerator::uniform_real(0.1, 0.2)), Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())),
	//		SurfaceProperty{Color::White}
	//		});
	//}

	//u32 i = 0;
	//for (const Vertex& vertex : geoSphereMesh0.getVertexArray())
	//{
	//	i++;
	//	const Vec3& position = vertex.position;

	//	const char* mesh = "box";
	//	const char* material = "diamond";
	//	const f32 scale = 0.1;

	//	std::string objectName = std::string("pole") + std::to_string(i);
	//	scene.addObject(Object{
	//		objectName + std::string("0"),
	//		mesh,
	//		material,
	//		Transform(position, Vec3(scale, 1000, scale)),
	//		SurfaceProperty{Color::White}
	//		});

	//	scene.addObject(Object{
	//		objectName + std::string("1"),
	//		mesh,
	//		material,
	//		Transform(position, Vec3(scale, scale, 1000)),
	//		SurfaceProperty{Color::White}
	//				});	
	//	
	//	scene.addObject(Object{
	//		objectName + std::string("2"),
	//		mesh,
	//		material,
	//		Transform(position, Vec3(1000, scale, scale)),
	//		SurfaceProperty{Color::White}
	//				});
	//}

	u32 i = 0;
	for (const Vertex& vertex : geoSphereMesh2.getVertexArray())
	{
		i++;
		const Vec3& position = vertex.position * 2.5;

		const char* mesh = "cylinder";
		const char* material = "fuzzyMetal";
		const f32 scale = 0.02;

		std::string objectName = std::string("pole") + std::to_string(i);
		scene.addObject(Object{
			objectName + std::string("0"),
			mesh,
			material,
			Transform(position, Vec3(scale, 1000, scale)),
			SurfaceProperty{Color::White}
			});

		scene.addObject(Object{
			objectName + std::string("1"),
			mesh,
			material,
			Transform(position, Vec3(scale, 1000, scale), Quaternion(M_PI / 2, Vec3::unitX())),
			SurfaceProperty{Color::White}
			});

		scene.addObject(Object{
			objectName + std::string("2"),
			mesh,
			material,
			Transform(position, Vec3(scale, 1000, scale), Quaternion(M_PI / 2, Vec3::unitZ())),
			SurfaceProperty{Color::White}
			});
	}

	for (u32 i = 0; i < 300; i++)
	{

		//scene.addObject(Object{
		//			std::to_string(i),
		//			"sphere",
		//			"highIntensityLight",
		//			Transform(Vec3::generateRandomUnitVector() * RandomGenerator::uniform_real(0.1, 20), Vec3(0.1, 0.1, 0.1), Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())),
		//			SurfaceProperty{Color::random()}
		//			});

		scene.addObject(Object{
					std::to_string(i),
					"plane",
					"highIntensityLight",
					Transform(Vec3::generateRandomUnitVector() * RandomGenerator::uniform_real(0.1, 20), Vec3::one() * 0.05, Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())),
					SurfaceProperty{Color::random()}
					});
	}
	//scene.addObject(Object{ "torus", "geoSphere2", "metal", Transform(Vec3(-3, 1, 0), 1), SurfaceProperty{Color::White} });
	//scene.addObject(Object{ "torus", "geoSphere0", "glass", Transform(Vec3(0, 1, -3), 1, Quaternion(0, Vec3(0,1,0))), SurfaceProperty{Color::White}});
	//scene.addObject(Object{ "torus", "cylinder", "fuzzyMetal", Transform(Vec3(0, 1, 3), 1, Quaternion(0, Vec3(0,1,0))), SurfaceProperty{Color::White} });
	//scene.addObject(Object{ "torus", "torus", "diamond", Transform(Vec3(0, 1, 0), 1, Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())), SurfaceProperty{Color::Red} });
	////scene.addObject(Object{ "torus", "geoSphere2", "glass", Transform(Vec3(3, 1, 0), -0.9), SurfaceProperty{Color::White} });
	//scene.addObject(Object{ "torus", "octahedron", "metal", Transform(Vec3(3, 1.2, 0), 1, Quaternion(RandomGenerator::uniform_real(0, 10),Vec3::generateRandomUnitVector())), SurfaceProperty{Color::Bronze} });

	//scene.addObject(Object{ "torus", "sphere", "highIntensityLight", Transform(Vec3(0, 1, 0), 0.6), SurfaceProperty{Color::White} });
	//
	//


	//Transform trans;
	//trans.setScaling(1000);
	//trans.setTranslation(Vec3(0, -1000, 0));
	//scene.addObject(Object{ "object1", "box", "diffuse", trans, SurfaceProperty{Color::Gray}});

	//const f32 skyScale = 10000;
	//trans.setScaling(skyScale, skyScale, skyScale);
	//trans.setTranslation(Vec3(0, 0, 0));
	//scene.addObject(Object{ "object1", "box", "lowIntesityLight", trans });

	//for (s32 a = -100; a < 11; a++)
	//{
	//	for (s32 b = -100; b < 11; b++)
	//	{
	//		f32 A = 1.5 * a;
	//		f32 B = 1.5 * b;
	//		Vec3 center{ A + 0.9f * RandomGenerator::uniform_real(), 0.3f, B + 0.9f * RandomGenerator::uniform_real() };
	//		

	//		Object object
	//		{
	//			std::string("obj") + std::to_string(a) + std::to_string(b),
	//			meshNameList[RandomGenerator::uniform_int(0, sizeof(meshNameList) / sizeof(meshNameList[0]))],
	//			 materialNameList[RandomGenerator::uniform_int(0, sizeof(materialNameList) / sizeof(materialNameList[0]))],
	//			 Transform(center, 0.3, Quaternion(RandomGenerator::uniform_real(0, 10), Vec3::generateRandomUnitVector())) ,
	//			 SurfaceProperty{ Color::random()}
	//		};
	//		scene.addObject(object);
	//	}
	//}

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