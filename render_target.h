#pragma once
#include <vector>
#include <fstream>
#include "Math/vector.h"
#include "color.h"
#include "util.h"

class RenderTarget
{
public:
	__host__ RenderTarget() = default;
	__host__ RenderTarget(u32 resolutionWidth, u32 resolutionHeight, Color clearColor = Color::White);
	
	__host__ RenderTarget& operator=(const RenderTarget& target);

	__device__ __host__ void setColor(const u32 pixelWidthIndex, const u32 pixelHeightIndex, const Color& color);

	__host__ void saveRenderResult(const std::string& path);

	u32 getResolutionWidth() const { return mResolutionWidth; }
	u32 getResolutionHeight() const { return mResolutionHeight; }
	u32 getPixelNum() const { return mPixelNum; }
	Color* getPixels() const {return mPixelArray;}

private:
	__device__ __host__ u32 calcIndex(u32 width, u32 height) const;

	const u32 MaxGradation = 255;

	u32 mResolutionWidth;
	u32 mResolutionHeight;
	u32 mPixelNum;
	Color* mPixelArray;
};