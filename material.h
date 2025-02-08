#pragma once
#include "ray.h"
#include "color.h"
#include "texture.h"

struct HitRecord;

class Material
{
public:
	/// <summary>
	/// ��{�I��true���Ԃ�B
	/// ��������̓��˂������Ȃ��ꍇ�͓��˕����ɂ����false�̏ꍇ���N����
	/// </summary>
	/// <returns></returns>
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) = 0;
};

class Lambertian : public Material
{
public:
	__device__ Lambertian(Color albedo) : albedo(albedo) {}
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	Color albedo;
};



class Metal : public Material
{
public:
	__device__ Metal(const Color& albedo, f32 fuzz = 0.0f) : albedo(albedo), fuzz(fuzz <= 1.0f ? fuzz : 1) {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


	Color albedo;
	f32 fuzz;
};


class Dielectric : public Material
{
public:
	__device__ Dielectric(float ref) : refIdx(ref) {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;

	__device__ static bool isRefract(const vec3& v, const vec3& n, float niOverNt, vec3& refracted);

	__device__ static f32 schlick(float cosine, float refIdx);


	f32 refIdx;
};

class Retroreflective : public Material
{
public:
	__device__ Retroreflective(const Color& albedo) : albedo(albedo) {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;

	Color albedo;
};

class SunLight : public Material
{
public:
	__device__ SunLight() {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;
};


class GravitationalField : public Material
{
public:
	__device__ GravitationalField(f32 gravityScale, vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	vec3 mCenter;
};

class QuasiGravitationalField : public Material
{
public:
	__device__ QuasiGravitationalField(f32 gravityScale, vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	vec3 mCenter;
};


class QuasiGravitationalField2 : public Material
{
public:
	__device__ QuasiGravitationalField2(f32 gravityScale, vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	vec3 mCenter;
};


class Rutherford : public Material
{
public:
	__device__ Rutherford(f32 gravityScale, vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	vec3 mCenter;
};


class QuasiRutherford : public Material
{
public:
	__device__ QuasiRutherford(f32 gravityScale, vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	vec3 mCenter;
};