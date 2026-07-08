#pragma once
#include "typeinfo.h"

namespace
{
	constexpr u32 RenderFrameCount = 300;


	const int xxx = 2;
	auto roundUp16 = [](int v) { return ((v + 15) / 16) * 16; };

	constexpr u32 pixelSizeHorizontal = roundUp16(3840 / xxx);
	constexpr u32 pixelSizeVertical = roundUp16(2160 / xxx);
	constexpr f32 aspectRatio = static_cast<f32>(pixelSizeHorizontal) / static_cast<f32>(pixelSizeVertical);
}