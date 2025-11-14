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

__global__ void setup_gpu()
{
	for (u32 i = 0; i < 32; i++)
	{
		curand_init(static_cast<unsigned long long>(i), 0, 0, &s[i]);
	}
}


Transform generateRandomTransform(const f32 scale = 100.0f)
{
	Transform transform;
	transform.setTranslation(Vec3(RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real()) * scale);
	transform.setRotation(Vec3(RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real()) * 10);
	transform.setScaling(1);
	return transform;
}


int main()
{
	ONCE_ON_GPU(setup_gpu)();

	Mesh sphereMesh = GeometryGenerator::sphereGenerator(100, 100);
	Mesh tetrahedronMesh = GeometryGenerator::tetrahedronGenerator();
	Mesh octahedronMesh = GeometryGenerator::octahedronGenerator();
	Mesh boxMesh = GeometryGenerator::boxGenerator();
	Mesh geoSphereMesh0 = GeometryGenerator::geoSphereGenerator(0);
	Mesh geoSphereMesh1 = GeometryGenerator::geoSphereGenerator(1);
	Mesh geoSphereMesh2 = GeometryGenerator::geoSphereGenerator(2);
	Mesh geoSphereMesh3 = GeometryGenerator::geoSphereGenerator(3);
	Mesh geoSphereMesh4 = GeometryGenerator::geoSphereGenerator(4);

	Material material0{Color::Gray, 0.0f, 0.0, 1.0f};
	Material material1{Color::Green, 0.0f, 0.0, 0.0f};
	Material material2{Color::Red, 0.0f, 0.0, 0.0f};
	Material material3{Color::Gray, 1.0f, 0.0, 0.0f};

	Scene scene;
	{
		scene.addMaterial("glass", material0);
		scene.addMaterial("metal", material0);
		scene.addMaterial("air", material1);
		scene.addMaterial("diamond", material2);
		scene.addMaterial("diffuse", material3);


		scene.addMesh("box", boxMesh);
		scene.addMesh("sphere", sphereMesh);
		scene.addMesh("octahedron", octahedronMesh);
		scene.addMesh("tetrahedron", tetrahedronMesh);
		scene.addMesh("geoSphere0", geoSphereMesh0);
		scene.addMesh("geoSphere1", geoSphereMesh1);
		scene.addMesh("geoSphere2", geoSphereMesh2);
		scene.addMesh("geoSphere3", geoSphereMesh3);
		scene.addMesh("geoSphere4", geoSphereMesh4);
	}

	Object object_tetra{"1", "tetrahedron", "diamond"};
	Object object_octa{"octa", "octahedron", "diamond"};
	Object object_sphere{"object0", "sphere", "glass"};
	Object object_box{"object1", "box", "diffuse" };
	Object object_geo1{"object2", "geoSphere1", "air", generateRandomTransform() };
	Object object_geo2{"object2_1", "geoSphere2", "air", generateRandomTransform()};
	Object object_geo3{"object2_1", "geoSphere3", "air", generateRandomTransform()};
	Object object_geo4{"object2_1", "geoSphere4", "air", generateRandomTransform()};

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
	};

	Result result;


	scene.addObject(object_geo1, Transform::translation(Vec3(0, 1, 0)));
	scene.addObject(object_geo2, Transform::translation(Vec3(-4, 1, 0)));
	scene.addObject(object_geo3, Transform::translation(Vec3(4, 1, 0)));

	Transform trans;
	trans.setScaling(1000);
	trans.setTranslation(Vec3(0, -1000, 0));
	scene.addObject(object_box, trans);

	for (s32 a = -11; a < 11; a++)
	{
		for (s32 b = -11; b < 11; b++)
		{
			Vec3 center{ a + 0.9f * RandomGenerator::uniform_real(), 0.2f, b + 0.9f * RandomGenerator::uniform_real() };
			
			Object object{
				(std::string("obj") + std::to_string(a) + std::to_string(b)).c_str(),
				meshNameList[RandomGenerator::uniform_int(0, sizeof(meshNameList) / sizeof(meshNameList[0]))] ,
				 "metal"};
			scene.addObject(object, Transform(center, 0.2));
		}
	}

	//Group group0("group0");
	//{
	//	result = group0.addChildObject(object0);
	//	result = group0.addChildObject(object1);
	//	result = group0.addChildObject(object1);
	//	result = group0.addChildObject(object2);
	//	result = group0.addChildObject(object2);
	//	result = group0.addChildObject(object2);
	//}


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