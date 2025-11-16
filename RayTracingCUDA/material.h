#pragma once
#include "color.h"



struct Material
{
	enum class MaterialType
	{
		LAMBERTIAN,
		METAL,
		DIELECTRIC,
		EMISSIVE
	};

	Material() = default;
	Material(MaterialType type, f32 roughness, f32 metallic, f32 ior, f32 transmission, const Color& emissionColor = Color(0x000000), bool isEmittable = false)
		: type(type)
		, roughness(roughness)
		, metallic(metallic)
		, ior(ior)
		, transmission(transmission)
		, emissionColor(emissionColor)
		, isEmittable(isEmittable)
	{ 
	}


	MaterialType type;

	f32 roughness;
	f32 metallic;
	f32 ior;
	f32 transmission = 0.0f;

	Color emissionColor;
	bool isEmittable;
};