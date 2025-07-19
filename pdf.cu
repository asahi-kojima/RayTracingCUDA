#include "pdf.h"


f32 CosinePdf::value(const Vec3& direction) const
{
    const f32 cos0 = Vec3::dot(direction, mOnb.getAxisZ());
    return (cos0 > 0 ? cos0 / M_PI : 0);
}

Vec3 CosinePdf::generateRandomDirection() const
{
    const f32 phi = RandomGeneratorGPU::uniform_real(0, 2 * M_PI);
    const f32 u = RandomGeneratorGPU::uniform_real();

    const f32 z = sqrtf(1 - u);
    const f32 x = cos(phi) * sqrtf(u);
    const f32 y = sin(phi) * sqrtf(u);

    const Vec3 v = mOnb.local(x, y, z);

    return v;
}