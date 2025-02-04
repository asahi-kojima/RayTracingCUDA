#pragma once
#include <vector>
#include <fstream>
#include "vector.h"
#include "color.h"
#include "util.h"

class RenderTarget
{
public:
	__host__ RenderTarget() = default;

	__host__ RenderTarget(u32 resolutionWidth, u32 resolutionHeight, Color clearColor = Color::White)
		: mResolutionWidth(resolutionWidth)
		, mResolutionHeight(resolutionHeight)
		, mPixelSize(resolutionHeight* resolutionWidth)
	{
		CHECK(cudaMallocManaged((void**)&mPixels, sizeof(Color) * mPixelSize));
		
		for (u32 i = 0; i < mPixelSize; i++)
		{
			mPixels[i] = clearColor;
		}
	}
	
	__host__ RenderTarget& operator=(const RenderTarget& target)
	{
		printf("RenderTarget Copied!\n");
		mResolutionHeight = target.mResolutionHeight;
		mResolutionWidth = target.mResolutionWidth;
		mPixelSize = target.mPixelSize;

		mPixels = target.mPixels;

		return *this;
	}

	__device__ __host__ void setColor(u32 width, u32 height, const Color& color)
	{
		u32 index = calcIndex(width, height);
		mPixels[index] = color;
	}

	__host__ void saveRenderResult(const std::string& path)
	{
		printf("save render result\n");
		std::ofstream stream(path.c_str());

		stream << "P3\n" << mResolutionWidth << " " << mResolutionHeight << "\n255\n";

		for (s32 j = mResolutionHeight - 1; j >= 0; j--)
		{
			for (u32 i = 0; i < mResolutionWidth; i++)
			{
				Color color = mPixels[calcIndex(i, j)];
				color = Color(sqrt(color[0]), sqrt(color[1]), sqrt(color[2]));
				color[0] = static_cast<f32>(static_cast<s32>(255.99 * color[0]));
				color[1] = static_cast<f32>(static_cast<s32>(255.99 * color[1]));
				color[2] = static_cast<f32>(static_cast<s32>(255.99 * color[2]));
				stream << color[0] << " " << color[1] << " " << color[2] << "\n";
			}
		}
		stream.close();
	}
	u32 getResolutionWidth() const { return mResolutionWidth; }
	u32 getResolutionHeight() const { return mResolutionHeight; }
	Color* getPixels() const {return mPixels;}

private:
	u32 mResolutionWidth;
	u32 mResolutionHeight;
	u32 mPixelSize;
	Color* mPixels;

	__device__ __host__ u32 calcIndex(u32 width, u32 height) const
	{
		return height * mResolutionWidth + width;
	}
};