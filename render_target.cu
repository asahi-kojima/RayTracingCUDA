#include "render_target.h"


RenderTarget::RenderTarget(u32 resolutionWidth, u32 resolutionHeight, Color clearColor)
    : mResolutionWidth(resolutionWidth)
    , mResolutionHeight(resolutionHeight)
    , mPixelNum(resolutionHeight* resolutionWidth)
{
    CHECK(cudaMallocManaged((void**)&mPixelArray, sizeof(Color) * mPixelNum));
    
    for (u32 i = 0; i < mPixelNum; i++)
    {
        mPixelArray[i] = clearColor;
    }
}

RenderTarget& RenderTarget::operator=(const RenderTarget& target)
{
    printf("RenderTarget Copied!\n");
    mResolutionHeight = target.mResolutionHeight;
    mResolutionWidth = target.mResolutionWidth;
    mPixelNum = target.mPixelNum;

    mPixelArray = target.mPixelArray;

    return *this;
}


void RenderTarget::setColor(const u32 pixelWidthIndex, const u32 pixelHeightIndex, const Color& color)
{
    u32 index = calcIndex(pixelWidthIndex, pixelHeightIndex);
    mPixelArray[index] = color;
}


void RenderTarget::saveRenderResult(const std::string& path)
{
    printf("save render result\n");
    std::ofstream stream(path.c_str());

    stream << "P3\n" << mResolutionWidth << " " << mResolutionHeight << "\n255\n";

    // f32 maxIntensity = 0.0f;
    // for (s32 j = mResolutionHeight - 1; j >= 0; j--)
    // {
    //     for (u32 i = 0; i < mResolutionWidth; i++)
    //     {
    //         Color color = mPixelArray[calcIndex(i, j)];
    //         maxIntensity = max(maxIntensity, color.r());
    //         maxIntensity = max(maxIntensity, color.g());
    //         maxIntensity = max(maxIntensity, color.b());
    //     }
    // }

    for (s32 j = mResolutionHeight - 1; j >= 0; j--)
    {
        for (u32 i = 0; i < mResolutionWidth; i++)
        {
            Color color = mPixelArray[calcIndex(i, j)];
            if (isnan(color[0]) || isnan(color[1]) || isnan(color[2]))
            {
                printf("=====================================");
                printf("nan detected!");
                printf("=====================================");
                //color = Color(0, 0, 0);
            }
            Color gammaCorrectedColor = Color(sqrt(color[0]), sqrt(color[1]), sqrt(color[2]));

            gammaCorrectedColor[0] = static_cast<f32>(static_cast<s32>(255.99 * gammaCorrectedColor[0]));
            gammaCorrectedColor[1] = static_cast<f32>(static_cast<s32>(255.99 * gammaCorrectedColor[1]));
            gammaCorrectedColor[2] = static_cast<f32>(static_cast<s32>(255.99 * gammaCorrectedColor[2]));

            stream << gammaCorrectedColor[0] << " " << gammaCorrectedColor[1] << " " << gammaCorrectedColor[2] << "\n";
        }
    }
    stream.close();
}


u32 RenderTarget::calcIndex(u32 width, u32 height) const
{
    return height * mResolutionWidth + width;
}