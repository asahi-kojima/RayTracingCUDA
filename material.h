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
	__device__ virtual Color emission(const f32 u, const f32 v, const Vec3& p) const { return Color(0,0,0); }
};

class Lambertian : public Material
{
public:
	__device__ Lambertian(Texture* texture) : mTexture(texture) {}
	__device__ Lambertian(Color color) : mTexture(new ConstantTexture(color)) {}
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;

private:
	Texture* mTexture;
};



class Metal : public Material
{
public:
	__device__ Metal(Texture* texture, f32 fuzz = 0.0f) : mTexture(texture), fuzz(fuzz <= 1.0f ? fuzz : 1) {}
	__device__ Metal(const Color& albedo, f32 fuzz = 0.0f) : mTexture(new ConstantTexture(albedo)), fuzz(fuzz <= 1.0f ? fuzz : 1) {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;

	Texture* mTexture;
	f32 fuzz;
};


class Dielectric : public Material
{
public:
	__device__ Dielectric(f32 ref, Color color = Color(0xFFFFFF)) : refIdx(ref), mGlassColor(color) {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;
	__device__ static f32 reflect_probability(float cosine, float refIdx);


	f32 refIdx;
	Color mGlassColor;
};

class BlackBody : public Material
{
public:
	__device__ BlackBody(){}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override
	{
		attenuation = Color(0x000000);
		return false;
	}
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
	__device__ SunLight(Color color, f32 intensity) :albedo(color), mIntensity(intensity) {}

private:
	Color albedo;
	f32 mIntensity;
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;
};

class DiffuseLight : public Material
{
public:
	__device__ DiffuseLight()  {}

private:
	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override {return false;}
	__device__ virtual Color emission(const f32 u, const f32 v, const Vec3& p) const override { return Color(1,1,1); }
};


class GravitationalField : public Material
{
public:
	__device__ GravitationalField(f32 gravityScale, Vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	Vec3 mCenter;
};

class QuasiGravitationalField : public Material
{
public:
	__device__ QuasiGravitationalField(f32 gravityScale, Vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	Vec3 mCenter;
};


class QuasiGravitationalField2 : public Material
{
public:
	__device__ QuasiGravitationalField2(f32 gravityScale, Vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	Vec3 mCenter;
};


class Rutherford : public Material
{
public:
	__device__ Rutherford(f32 gravityScale, Vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	Vec3 mCenter;
};


class QuasiRutherford : public Material
{
public:
	__device__ QuasiRutherford(f32 gravityScale, Vec3 center) :mGravityScale(gravityScale), mCenter(center) {}

	__device__ virtual bool scatter(const Ray& ray_in, const HitRecord& record, Color& attenuation, Ray& ray_scattered) override;


private:
	f32 mGravityScale;
	constexpr static f32 G = 1.0f;
	Vec3 mCenter;
};