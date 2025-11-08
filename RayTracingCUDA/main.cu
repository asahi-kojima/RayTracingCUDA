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

struct DeviceInstanceRecord
{
	Mat4 transform;
	Mat4 invTransform;
	AABB aabbInWorld;

	u32 blasIndex;

	u32 vertexOffset;
	u32 indexOffset;

	u32 materialID;
};

struct BVHNode
{
	AABB aabb;
};

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
	Mesh sphereMesh = GeometryGenerator::sphereGenerator();
	Mesh boxMesh = GeometryGenerator::boxGenerator();
	Mesh geoSphereMesh = GeometryGenerator::geoSphereGenerator(2);

	Material material0{Color::Bronze, 1.0f, 0.0};
	Material material1{Color::Green, 1.0f, 0.0};

	Scene scene;
	{
		scene.addMaterial("glass", material0);
		scene.addMaterial("metal", material0);
		scene.addMaterial("air", material1);


		scene.addMesh("sphere", sphereMesh);
		scene.addMesh("box", boxMesh);
		scene.addMesh("geoSphere", geoSphereMesh);
	}

	Object object0{"object0", "sphere", "glass"};
	Object object1{"object1", "box", "metal"};
	Object object2{"object2", "geoSphere", "air"};
	Object object3{"object2", "geoSphere", "air", generateRandomTransform() };
	Object object4{"object2_1", "geoSphere", "air", generateRandomTransform()};

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
		result = group3.addChildObject(object2);
		result = group3.addChildObject(object3);
		result = group3.addChildObject(object4);
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
	}



}