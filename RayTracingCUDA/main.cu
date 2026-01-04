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
#include <sstream>
#include <iomanip>
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



// クーロン力計算
Vec3 calcCoulombForce(const Vec3& posA, f32 chargeA, const Vec3& posB, f32 chargeB) {
	constexpr f32 k = 8.9875517923e9f; // クーロン定数
	Vec3 r =  posA - posB;
	f32 dist2 = r.lengthSquared();
	if (dist2 < 1e-10f) return Vec3::zero();
	return r.normalize() * (k * chargeA * chargeB / dist2);
}

// 質点Aの位置を計算
Vec3 calcPositionAfterT(
	Vec3 posA, f32 massA, f32 chargeA,
	Vec3 posB, f32 massB, f32 chargeB,
	f32 T, f32 dt = 1e-4f
)
{
	Vec3 velA = Vec3::zero();
	for (f32 t = 0; t < T; t += dt)
	{
		Vec3 force = calcCoulombForce(posA, chargeA, posB, chargeB);
		Vec3 accA = force / massA;
		velA += accA * dt;
		posA += velA * dt;
	}
	return posA;
}

//// 使用例
//void example() {
//	Vec3 posA(0, 0, 0);      // 質点Aの初期位置
//	f32 massA = 1.0f;        // 質点Aの質量
//	f32 chargeA = 1.0e-6f;   // 質点Aの電荷
//	Vec3 posB(10, 0, 0);     // 粒子Bの初期位置
//	Vec3 velB(-1, 0, 0);     // 粒子Bの初速度
//	f32 massB = 1.0f;        // 粒子Bの質量
//	f32 chargeB = 1.0e-6f;   // 粒子Bの電荷
//	f32 T = 1.0f;            // 計算する時刻
//	Vec3 result = calcPositionAfterT(posA, massA, chargeA, posB, velB, massB, chargeB, T);
//	result.debugPrint("Aの位置");
//}
//


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
	Material light{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::White * 1, true};
	Material lowIntesityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::White * 0.1, true};
	Material highIntensityLight{Material::MaterialType::EMISSIVE, 1.0f, 0.0, 0.0f, 0.0f, Color::White * 10, true};

	const f32 time = 5;
	const s32 imageNum = time * 24;
	const f32 fps = imageNum / time;
	for (u32 loop = 1; loop < imageNum; loop++)
	{
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
			"highIntensityLight"
		};

		Result result;



		std::string groupName = "sceneGroup_" + std::to_string(loop);
		Group particles(groupName.c_str());
		{
			const u32 particleCount = 1000 *10;

			const f32 mass = 1.0f;
			const f32 charge = 0.0001;
			const f32 T = 5.0f;
			const f32 t = T * (loop * 1.0f / imageNum);
			const f32 dt = 1e-4f;

			const f32 radius = 3.0;

			const Vec3 initPosB = Vec3(0, 0, 4);
			const Vec3 finalPosB = Vec3(0, 0, -4);
			const Vec3 velB = (finalPosB - initPosB) / T;

			const Vec3 posB = initPosB + velB * t;
			for (u32 i = 0; i < particleCount; i++)
			{
				// 初期位置と速度をランダムに生成
				Vec3 posA = Vec3::generateRandomUnitVector() * radius * RandomGenerator::uniform_real(0.2f, 1.0f);

				// 質点Aの位置を計算
				Vec3 newPosA = calcPositionAfterT(posA, mass, charge, posB, mass, charge, t, dt);


				const std::string meshName = meshNameList[RandomGenerator::uniform_int(0, sizeof(meshNameList) / sizeof(meshNameList[0]) - 1)];
				const std::string materialName = materialNameList[RandomGenerator::uniform_int(0, sizeof(materialNameList) / sizeof(materialNameList[0]) - 1)];
				result = particles.addChildObject(Object{
					"particle_" + std::to_string(i),
					"geoSphere1",
					"diffuse",
					Transform(newPosA, Vec3::one() * 0.06, Quaternion(0, Vec3::unitZ())),
					SurfaceProperty{Color::Bronze} });
			}

			result = particles.addChildObject(Object{
				"initparticle",
				"geoSphere2",
				"fuzzyMetal",
				Transform(posB, Vec3::one() * 1, Quaternion(0, Vec3::unitZ())),
				SurfaceProperty{Color::Blue} });


			result = particles.addChildObject(Object{
				"light" ,
				"box",
				"light",
				Transform(Vec3::zero(), Vec3::one() * 100, Quaternion(0, Vec3::unitZ())),
				SurfaceProperty{Color::White} });

		}

		scene.addGroup(particles);



		result = scene.build();
		result = scene.initLaunchParams();

		std::ostringstream oss;
		oss << std::setw(3) << std::setfill('0') << loop;
		std::string filename = oss.str() + "_renderResult.ppm";
		result = scene.render(filename);
		cudaDeviceSynchronize();
	}
}