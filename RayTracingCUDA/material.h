#pragma once
#include "color.h"

struct Material
{
	Color albedo;
	f32 roughness;
	f32 specular;
	f32 diffuse;
};