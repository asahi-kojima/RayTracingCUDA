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


Transform generateRandomTransform(const f32 scale = 1.0f)
{
	Transform transform;
	transform.setTranslation(Vec3(RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real()) * scale);
	transform.setRotation(Vec3(RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real(), RandomGenerator::signed_uniform_real()) * 10);
	transform.setScaling(scale * 0.1);
	return transform;
}


int main()
{
	ONCE_ON_GPU(setup_gpu)();

	Mesh sphereMesh = GeometryGenerator::geoSphereGenerator(1);
	Mesh boxMesh = GeometryGenerator::boxGenerator();
	Mesh geoSphereMesh2 = GeometryGenerator::geoSphereGenerator(2);
	Mesh geoSphereMesh3 = GeometryGenerator::geoSphereGenerator(3);
	Mesh geoSphereMesh4 = GeometryGenerator::geoSphereGenerator(4);

	Material material0{Color::Bronze, 1.0f, 0.0, 1.0f};
	Material material1{Color::Green, 1.0f, 0.0, 0.0f};

	Scene scene;
	{
		scene.addMaterial("glass", material0);
		scene.addMaterial("metal", material0);
		scene.addMaterial("air", material1);


		scene.addMesh("sphere", sphereMesh);
		scene.addMesh("box", boxMesh);
		scene.addMesh("geoSphere2", geoSphereMesh2);
		scene.addMesh("geoSphere3", geoSphereMesh3);
		scene.addMesh("geoSphere4", geoSphereMesh4);
	}

	Object object0{"object0", "sphere", "glass"};
	Object object1{"object1", "box", "metal"};
	Object object2{"object2", "geoSphere2", "air"};
	Object object3{"object2", "geoSphere2", "air", generateRandomTransform() };
	Object object4{"object2_1", "geoSphere2", "air", generateRandomTransform()};

	Object object5{"object2_1", "geoSphere3", "air", generateRandomTransform()};
	Object object6{"object2_1", "geoSphere3", "air", generateRandomTransform()};
	Object object7{"object2_1", "geoSphere3", "metal", generateRandomTransform()};

	Result result;
	Group group0("group0");
	{
		result = group0.addChildObject(object0);
		result = group0.addChildObject(object1);
		result = group0.addChildObject(object1);
		result = group0.addChildObject(object2);
		result = group0.addChildObject(object2);
		result = group0.addChildObject(object2);
	}

	Group group1("group1", generateRandomTransform());
	{
		result = group1.addChildGroup(group0);
		result = group1.addChildObject(object0);
		result = group1.addChildObject(object1);
		result = group1.addChildObject(object1);
		result = group1.addChildObject(object3);
	}

	Group group2("group2");
	{
		result = group2.addChildObject(object2);
		result = group2.addChildObject(object3);
		result = group2.addChildObject(object4);
	}
	
	Group group3("group3", generateRandomTransform());
	{
		result = group3.addChildObject(object5);
		result = group3.addChildObject(object6);
		result = group3.addChildObject(object7);
	}

	
	{
		result = scene.addObject(object0);
		result = scene.addObject(object1);
		result = scene.addObject(object2);
		result = scene.addObject(object3, generateRandomTransform());

		result = scene.addGroup(group0);
		result = scene.addGroup(group1);
		result = scene.addGroup(group2, generateRandomTransform());
		result = scene.addGroup(group3, generateRandomTransform());
		result = scene.addGroup(group3, generateRandomTransform());
		result = scene.addGroup(group3, generateRandomTransform());
		result = scene.addGroup(group0);
	}

	for (u32 i = 0; i < 100; i++)
	{
		result = scene.addObject(object0,generateRandomTransform(), object0.getName() + std::to_string(i));
		result = scene.addObject(object1,generateRandomTransform(), object1.getName() + std::to_string(i));
		result = scene.addObject(object2,generateRandomTransform(), object2.getName() + std::to_string(i));
		result = scene.addObject(object3, generateRandomTransform(), object3.getName() + std::to_string(i));

		result = scene.addGroup(group0, generateRandomTransform(), object0.getName() + std::to_string(i));
		result = scene.addGroup(group1, generateRandomTransform(), group1.getName() + std::to_string(i));
		result = scene.addGroup(group2, generateRandomTransform(), group2.getName() + std::to_string(i));
		result = scene.addGroup(group3, generateRandomTransform(), group3.getName() + std::to_string(i));
		result = scene.addGroup(group3, generateRandomTransform(), group3.getName() + std::to_string(i));
		result = scene.addGroup(group3, generateRandomTransform(), group3.getName() + std::to_string(i));
		result = scene.addGroup(group0, generateRandomTransform(), object0.getName() + std::to_string(i));
	}


	result = scene.build();
	result = scene.initLaunchParams();
	result = scene.render();
	//cudaDeviceSynchronize();
}