#include "material.h"
#include "Object/hittable.h"
#include "util.h"
#include "Object/object.h"
//======================================================
// ランバート
//======================================================
bool Lambertian::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	if (Vec3::dot(ray_in.direction(), record.normal) > 0)
	{
		return false;
	}
	const Vec3 target = record.position + record.normal + Vec3::generateRandomUnitVector();
	ray_scattered.direction() = target - record.position;
	ray_scattered.origin() = record.position;
	attenuation = record.hitObject->getSurfaceProperty().getAlbedo();
	pdf = Vec3::dot(record.normal, ray_scattered.direction()) / M_PI;
	isDiffuse = true;
	return true;
}

f32 Lambertian::scatteringPdf(const Ray& incidentRay, const HitRecord& record, const Ray& scatterdRay) const
{
	const f32 cosine = Vec3::dot(record.normal, scatterdRay.direction().normalize());
	return cosine > 0 ? cosine / M_PI : 0;
}


//======================================================
// 金属
//======================================================
bool Metal::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	Vec3 reflected_ray = Vec3::reflect(ray_in.direction(), record.normal);
	ray_scattered = Ray(record.position, reflected_ray + fuzz * Vec3::generateRandomUnitVector() * 0.1f);
	
	if (Vec3::dot(ray_scattered.direction(), record.normal) > 0)
	{
		attenuation = record.hitObject->getSurfaceProperty().getAlbedo();//mTexture->color(0, 0, record.position);
		return true;
	}

	return false;
}

//======================================================
// 誘電体
//======================================================
bool Dielectric::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	attenuation = mGlassColor;

	const Vec3& position = record.position;
	const Vec3& normal = record.normal;
	const Vec3 direction = ray_in.direction().normalize();

	const f32 signed_cos_theta = clamp<f32>(Vec3::dot(normal, direction), -1.0f, 1.0f);//cos_between_normal_and_direction;
	const f32 cos_theta = fabsf(signed_cos_theta);
	const f32 sin_theta = sqrt(1.0f - cos_theta * cos_theta + (1e-5));
	

	//cosの値が負の時は空気中から入射していることになる。
	//逆に正の場合は媒質中から飛び出そうとしている状況
	const bool isFromOutside = (signed_cos_theta < 0);

	//相対屈折率
	const f32 ni_over_nt = (isFromOutside ? 1.0f / refIdx : refIdx);

	//レイに対する外向き法線
	const Vec3 outword_normal = normal * (isFromOutside ? 1 : -1);

	//全反射もしくは屈折出来るが反射が起きる場合は反射処理
	if (ni_over_nt * sin_theta >= 1 || (RandomGeneratorGPU::uniform_real() < reflect_probability(cos_theta, ni_over_nt)))
	{
		const Vec3 reflected_ray_direction = Vec3::reflect(direction, outword_normal);
		ray_scattered = Ray(position, reflected_ray_direction);
	}
	//屈折する場合
	else
	{
		const Vec3 refracted_ray_direction = -sqrt(1 - ni_over_nt * ni_over_nt * sin_theta * sin_theta + (1e-7)) * outword_normal + ni_over_nt * (direction + cos_theta * outword_normal);
		ray_scattered = Ray(position, refracted_ray_direction);
	}

	return true;
}

f32 Dielectric::reflect_probability(f32 cosine, f32 refIdx)
{
	f32 r0 = (1.0f - refIdx) / (1.0f + refIdx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}

//======================================================
// 再帰性反射素材
//======================================================
bool Retroreflective::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	ray_scattered.direction() = -ray_in.direction();
	ray_scattered.origin() = record.position;

	attenuation = albedo;

	return false;
}

//======================================================
// 光源
//======================================================

bool SunLight::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	attenuation = albedo * mIntensity;
	return false;
}

//======================================================
// 重力場
//======================================================
bool GravitationalField::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	const f32 M = mGravityScale;
	const f32 m = 1.0f;
	const f32 v = 10.0f;

	const Vec3 CP = record.position - mCenter;
	const f32 R = CP.length();

	const Vec3 OC = mCenter - ray_in.origin();
	const Vec3 D = ray_in.direction();
	const f32 ray_center_dist_squared = (D * (Vec3::dot(OC, D) / D.lengthSquared()) - OC).lengthSquared();

	const f32 E = 0.5f * m * v * v - G * M * m / R;
	const f32 L_squared = m * m * v * v * ray_center_dist_squared;
	const f32 R0 = L_squared / (G * M);
	const f32 typical_E = L_squared / (2.0f * R0 * R0); // 典型的なエネルギースケールを意味しており、実際のエネルギーとは別

	// 離心率
	const f32 e = sqrtf(1.0f + E / typical_E);
	if (e < 1.0f)
	{
		attenuation = Color(0x000000);
		return false;
	}

	attenuation = Color(0xFFFFFF);

	{
		const Vec3 ux = -Vec3::normalize(D);
		const Vec3 uz = Vec3::normalize(Vec3::cross(ux, CP));
		const Vec3 uy = Vec3::cross(uz, ux);
		const f32 h = sqrtf(ray_center_dist_squared);
		const f32 theta = asinf(h / R);

		const f32 phi = -(acosf(((R0 / OC.length()) - 1.0f) / e) - theta); // assert(phi < 0);
		const f32 phi2 = 2.0f * phi;

		const f32 x = R * cos(theta);
		const f32 y = R * sin(theta);
		const f32 cosPhi2 = cos(phi2);
		const f32 sinPhi2 = sin(phi2);

		const f32 outgoing_x = cosPhi2 * x + sinPhi2 * y;
		const f32 outgoing_y = sinPhi2 * x - cosPhi2 * y;
		const Vec3 outgoing_position = outgoing_x * ux + outgoing_y * uy + mCenter;
		const Vec3 outgoing_dir = cosPhi2 * ux + sinPhi2 * uy;

		ray_scattered.direction() = outgoing_dir;
		ray_scattered.origin() = outgoing_position;
	}
	return true;
}

//======================================================
// 疑似重力場（敢えて計算ミスを入れている）
//======================================================
bool QuasiGravitationalField::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	const f32 M = mGravityScale;
	const f32 m = 1.0f;
	const f32 v = 10.0f;

	const Vec3 CP = record.position - mCenter;
	const f32 R = CP.length();

	const Vec3 OC = mCenter - ray_in.origin();
	const Vec3 D = ray_in.direction();
	const f32 ray_center_dist = (D * (Vec3::dot(OC, D) / D.lengthSquared()) - OC).length();

	const f32 E = 0.5f * m * v * v - G * M * m / R;
	const f32 L = m * v * ray_center_dist;
	const f32 R0 = L * L / (G * M);
	const f32 typical_E = L * L / (2.0f * R0 * R0); // 典型的なエネルギースケールを意味しており、実際のエネルギーとは別

	// 離心率
	const f32 e = sqrtf(1.0f + E / typical_E);
	if (e < 1.0f)
	{
		attenuation = Color(0x000000);
		return false;
	}

	attenuation = Color(0xFFFFFF);

	{
		const Vec3 ux = -Vec3::normalize(D);
		const Vec3 uz = Vec3::normalize(Vec3::cross(ux, CP));
		const Vec3 uy = Vec3::cross(uz, ux);
		const f32 h = abs(Vec3::dot(ux, CP));
		const f32 theta = asinf(h / R);

		const f32 phi = -(acosf(((R0 / OC.length()) - 1.0f) / e) - theta);
		const f32 phi2 = 2.0f * phi;

		const f32 x = R * cos(theta);
		const f32 y = R * sin(theta);
		const f32 cosPhi2 = cos(phi2);
		const f32 sinPhi2 = sin(phi2);

		const f32 outgoing_x = cosPhi2 * x + sinPhi2 * y;
		const f32 outgoing_y = sinPhi2 * x - cosPhi2 * y;
		const Vec3 outgoing_position = outgoing_x * ux + outgoing_y * uy + mCenter;
		const Vec3 outgoing_dir = cosPhi2 * ux + sinPhi2 * uy;

		ray_scattered.direction() = outgoing_dir;
		ray_scattered.origin() = outgoing_position;
	}
	return true;
}

//======================================================
// 疑似重力場2（敢えて計算ミスを入れている）
//======================================================
bool QuasiGravitationalField2::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	const f32 M = mGravityScale;
	const f32 m = 1.0f;
	const f32 v = 10.0f;

	const Vec3 CP = record.position - mCenter;
	const f32 R = CP.length();

	const Vec3 OC = mCenter - ray_in.origin();
	const Vec3 D = ray_in.direction();
	const f32 ray_center_dist = (D * (Vec3::dot(OC, D) / D.lengthSquared()) - OC).length();

	const f32 E = 0.5f * m * v * v - G * M * m / R;
	const f32 L = m * v * ray_center_dist;
	const f32 R0 = L * L / (G * M);
	const f32 typical_E = L * L / (2.0f * R0 * R0); // 典型的なエネルギースケールを意味しており、実際のエネルギーとは別

	// 離心率
	const f32 e = sqrtf(1.0f + E / typical_E);
	if (e < 1.0f)
	{
		attenuation = Color(0x000000);
		return false;
	}

	attenuation = Color(0xFFFFFF);

	{
		const Vec3 ux = -Vec3::normalize(D);
		const Vec3 uz = Vec3::normalize(Vec3::cross(ux, CP));
		const Vec3 uy = Vec3::cross(uz, ux);
		const f32 h = abs(Vec3::dot(uz, CP));
		const f32 theta = asinf(h / R);

		const f32 phi = -(acosf(((R0 / OC.length()) - 1.0f) / e) - theta);
		const f32 phi2 = 2.0f * phi;

		const f32 x = R * cos(theta);
		const f32 y = R * sin(theta);
		const f32 cosPhi2 = cos(phi2);
		const f32 sinPhi2 = sin(phi2);

		const f32 outgoing_x = cosPhi2 * x + sinPhi2 * y;
		const f32 outgoing_y = sinPhi2 * x - cosPhi2 * y;
		const Vec3 outgoing_position = outgoing_x * ux + outgoing_y * uy + mCenter;
		const Vec3 outgoing_dir = cosPhi2 * ux + sinPhi2 * uy;

		ray_scattered.direction() = outgoing_dir;
		ray_scattered.origin() = outgoing_position;
	}
	return true;
}

//======================================================
// ラザフォード散乱
//======================================================
bool Rutherford::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	const f32 M = mGravityScale;
	const f32 m = 1.0f;
	const f32 v = 10.0f;

	const Vec3 CP = record.position - mCenter;
	const f32 R = CP.length();

	const Vec3 OC = mCenter - ray_in.origin();
	const Vec3 D = ray_in.direction();
	const f32 ray_center_dist = (D * (Vec3::dot(OC, D) / D.lengthSquared()) - OC).length();

	const f32 E = 0.5f * m * v * v + G * M * m / R;
	const f32 L = m * v * ray_center_dist;
	const f32 R0 = L * L / (G * M);
	const f32 typical_E = L * L / (2.0f * R0 * R0); // 典型的なエネルギースケールを意味しており、実際のエネルギーとは別

	// 離心率
	const f32 e = sqrtf(1.0f + E / typical_E);

	attenuation = Color(0xFFFFFF);

	{
		const Vec3 ux = -Vec3::normalize(D);
		const Vec3 uz = Vec3::normalize(Vec3::cross(ux, CP));
		const Vec3 uy = Vec3::cross(uz, ux);
		const f32 h = ray_center_dist;
		const f32 theta = asinf(h / R);

		const f32 phi = 2.0f * atan(G * M / (h * v * v));

		const f32 cosTheta = cos(theta);
		const f32 sinTheta = sin(theta);
		const f32 cosPhi = cos(phi);
		const f32 sinPhi = sin(phi);

		const f32 x = R * cosTheta;
		const f32 y = R * sinTheta;

		const f32 cosPhiTheta = cosPhi * cosTheta - sinPhi * sinTheta; // cos(phi + theta);
		const f32 sinPhiTheta = sinPhi * cosTheta + cosPhi * sinTheta; // sin(phi + theta);

		const f32 outgoing_x = -cosPhiTheta * x - sinPhiTheta * y; //-cos(phi + theta) * x + sin(phi + theta) * y;
		const f32 outgoing_y = sinPhiTheta * x - cosPhiTheta * y;  //-sin(phi + theta) * x - cos(phi + theta) * y;
		const Vec3 outgoing_position = outgoing_x * ux + outgoing_y * uy + mCenter;
		const Vec3 outgoing_dir = -cosPhi * ux + sinPhi * uy;

		ray_scattered.direction() = outgoing_dir;
		ray_scattered.origin() = outgoing_position;
	}
	return true;
}

//======================================================
// 疑似ラザフォード散乱
//======================================================
bool QuasiRutherford::scatter(const Ray &ray_in, const HitRecord &record, Color &attenuation, Ray &ray_scattered, f32& pdf,  bool& isDiffuse)
{
	const f32 M = mGravityScale;
	const f32 m = 1.0f;
	const f32 v = 10.0f;

	const Vec3 CP = record.position - mCenter;
	const f32 R = CP.length();

	const Vec3 OC = mCenter - ray_in.origin();
	const Vec3 D = ray_in.direction();
	const f32 ray_center_dist = (D * (Vec3::dot(OC, D) / D.lengthSquared()) - OC).length();

	const f32 E = 0.5f * m * v * v + G * M * m / R;
	const f32 L = m * v * ray_center_dist;
	const f32 R0 = L * L / (G * M);
	const f32 typical_E = L * L / (2.0f * R0 * R0); // 典型的なエネルギースケールを意味しており、実際のエネルギーとは別

	// 離心率
	const f32 e = sqrtf(1.0f + E / typical_E);

	attenuation = Color(0xFFFFFF);

	{
		const Vec3 ux = -Vec3::normalize(D);
		const Vec3 uz = Vec3::normalize(Vec3::cross(ux, CP));
		const Vec3 uy = Vec3::cross(uz, ux);
		const f32 h = ray_center_dist;
		const f32 theta = asinf(h / R);

		const f32 phi = 2.0f * atan(G * M / (h * v * v));

		const f32 x = R * cos(theta);
		const f32 y = R * sin(theta);

		const f32 outgoing_x = -(cos(phi + theta) * x + sin(phi + theta) * y);
		const f32 outgoing_y = -(sin(phi + theta) * x - cos(phi + theta) * y);
		const Vec3 outgoing_position = outgoing_x * ux + outgoing_y * uy + mCenter;
		const Vec3 outgoing_dir = -cos(phi) * ux + sin(phi) * uy;

		ray_scattered.direction() = outgoing_dir;
		ray_scattered.origin() = outgoing_position;
	}
	return true;
}
