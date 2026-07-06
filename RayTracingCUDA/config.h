#pragma once
#include "typeinfo.h"

namespace
{
	const int xxx = 1;
	auto roundUp16 = [](int v) { return ((v + 15) / 16) * 16; };
}

namespace
{
	constexpr u32 RenderFrameCount = 500;


	constexpr u32 pixelSizeHorizontal = roundUp16(2000 / xxx);
	constexpr u32 pixelSizeVertical = roundUp16(2000 / xxx);
	constexpr f32 aspectRatio = static_cast<f32>(pixelSizeHorizontal) / static_cast<f32>(pixelSizeVertical);
}