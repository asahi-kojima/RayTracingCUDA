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
	Mesh torusMesh       = GeometryGenerator::torusGenerator(0.05f, 40, 20);

	Material pureMetal{Material::MaterialType::METAL, 0.0f, 1.0, 1.0f, 0.0f};
	Material fuzzyMetal{Material::MaterialType::METAL, 0.2f, 0.0, 0.0f, 0.0f};
	Material water{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.1f, 0.0f};
	Material glass{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 1.5f, 0.0f};
	Material diamond{Material::MaterialType::DIELECTRIC, 0.0f, 0.0, 2.5f, 0.0f};
	Material pureLambertian{Material::MaterialType::LAMBERTIAN, 1.0f, 0.0, 0.0f, 0.0f};
	Material light{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 1, true};
	Material lowIntesityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 0.1, true};
	Material highIntensityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::Azure * 10, true};

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
		const f32 boardScale = 555.0f;
		{
			result = cornellBox.addChildObject(Object{
				"Ceiling",
				"plane",
				"diffuse",
				Transform(Vec3(boardScale / 2, boardScale, boardScale / 2), Vec3(boardScale / 2, 1, boardScale / 2), Quaternion(M_PI, Vec3::unitZ())),
				SurfaceProperty{Color::White} });
		}
		{
			result = cornellBox.addChildObject(Object{
				"Floor",
				"plane",
				"diffuse",
				Transform(Vec3(boardScale / 2, 0, boardScale / 2), Vec3(boardScale / 2, 1, boardScale / 2)),
				SurfaceProperty{Color::White} });
		}
		{
			result = cornellBox.addChildObject(Object{
				"BackWall",
				"plane",
				"diffuse",
				Transform(Vec3(boardScale / 2, boardScale / 2, boardScale), Vec3(boardScale, 1, boardScale), Quaternion(3 * M_PI / 2, Vec3::unitX())),
				SurfaceProperty{Color::White} });
		}
		{
			result = cornellBox.addChildObject(Object{
				"LeftWall",
				"plane",
				"diffuse",
				Transform(Vec3(boardScale, boardScale / 2, boardScale / 2), Vec3(boardScale / 2, 1, boardScale / 2), Quaternion(M_PI / 2, Vec3::unitZ())),
				SurfaceProperty{Color::Red} });
		}
		{
			result = cornellBox.addChildObject(Object{
				"RightWall",
				"plane",
				"diffuse",
				Transform(Vec3(0, boardScale / 2, boardScale / 2), Vec3(boardScale / 2, 1, boardScale / 2), Quaternion(-M_PI / 2, Vec3::unitZ())),
				SurfaceProperty{Color::Green} });
		}

		constexpr f32 BoxScale = 165.0f;
		{
			const f32 angle = -M_PI / 10;
			const f32 z = (cosf(angle) - sinf(angle)) * (BoxScale / 2);
			const f32 x = (sinf(angle) + cosf(angle)) * (BoxScale / 2);
			const Vec3 position(x + 130, BoxScale / 2, z + 65);

			result = cornellBox.addChildObject(Object{
				"RightBox",
				"box",
				"metal",
				Transform(position, Vec3(BoxScale,BoxScale,BoxScale) * 0.5f, Quaternion(angle, Vec3::unitY())),
				SurfaceProperty{Color::Silver} });
		}

		{
			const f32 angle = M_PI / 12;
			const f32 z = (cosf(angle) - sinf(angle)) * (BoxScale / 2);
			const f32 x = (sinf(angle) + cosf(angle)) * (BoxScale / 2);
			const Vec3 position(x + 265, 2 * BoxScale / 2, z + 295);

			result = cornellBox.addChildObject(Object{
				"RightBox",
				"box",
				"metal",
				Transform(position, Vec3(BoxScale,2 * BoxScale,BoxScale) * 0.5f, Quaternion(angle, Vec3::unitY())),
				SurfaceProperty{Color::Silver} });
		}

		{
			constexpr f32 LightSizeScale = 0.3f;
			result = cornellBox.addChildObject(Object{
				"Light",
				"box",
				"highIntensityLight",
				Transform(Vec3(555 / 2, 554, 555 / 2), Vec3(555 * LightSizeScale, 1.0, 555 * LightSizeScale) * 0.5, Quaternion(-M_PI, Vec3::unitZ())),
				SurfaceProperty{Color::Bronze} });
		}
	}
	
	scene.addGroup(cornellBox);



	result = scene.build();
	result = scene.initLaunchParams();
	result = scene.render();
	cudaDeviceSynchronize();
}