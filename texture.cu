#include "texture.h"

Color ConstantTexture::color(f32 u, f32 v, const vec3& p)
{
    return mColor;
}

//Color CheckerTexture::color(f32 u, f32 v, const vec3& p)
//{
//    const f32 sine_product = sin(10 * p[0]) * sin(10 * p[1]) * sin(10 * p[2]);
//    if (sine_product > 0)
//    {
//        return mEvenTexture->color(u, v, p);
//    }
//    else
//    {
//        return mOddTexture->color(u, v, p);
//    }
//}
