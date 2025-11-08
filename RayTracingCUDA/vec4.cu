#include <cassert>
#include <math.h>
#include <random>
#include "vector.h"
//#include "util.h"


f32& Vec4::operator[](size_t i)
{
#ifdef DEBUG
    if (i >= 4 || i < 0)
    {
        assert(0);
    }
#endif
    return mElements[i];
}

f32 Vec4::operator[](size_t i) const
{
#ifdef DEBUG
    if (i >= 4 || i < 0)
    {
        assert(0);
    }
#endif
    return mElements[i];
}

Vec4  Vec4::operator+(const Vec4& v) const
{
    const Vec4& u = *this;
    f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = u[i] + v[i];
    }
    return Vec4(result);
}

Vec4& Vec4::operator+=(const Vec4& v)
{
    for (u32 i = 0; i < 4; i++)
    {
        (*this)[i] += v[i];
    }
    return *this;
}

Vec4  Vec4::operator+(const f32 v) const
{
    const Vec4& u = *this;
    f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = u[i] + v;
    }
    return Vec4(result);
}

Vec4& Vec4::operator+=(const f32 v)
{
    for (u32 i = 0; i < 4; i++)
    {
        (*this)[i] += v;
    }
    return *this;
}

Vec4  Vec4::operator-(const Vec4& v) const
{
    const Vec4& u = *this;
    f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = u[i] - v[i];
    }
    return Vec4(result);
}

Vec4& Vec4::operator-=(const Vec4& v)
{
    for (u32 i = 0; i < 4; i++)
    {
        (*this)[i] -= v[i];
    }
    return *this;
}

Vec4  Vec4::operator-(const f32 v) const
{
    const Vec4& u = *this;
    f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = u[i] - v;
    }
    return Vec4(result);
}

Vec4& Vec4::operator-=(const f32 v)
{
    for (u32 i = 0; i < 4; i++)
    {
        (*this)[i] -= v;
    }
    return *this;
}

Vec4  Vec4::operator*(const Vec4& v) const
{
    const Vec4& u = *this;
    f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = u[i] * v[i];
    }
    return Vec4(result);
}

Vec4& Vec4::operator*=(const Vec4& v)
{
    for (u32 i = 0; i < 4; i++)
    {
        (*this)[i] *= v[i];
    }
    return *this;
}

Vec4  Vec4::operator*(const f32 v) const
{
    const Vec4& u = *this;
    f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = u[i] * v;
    }
    return Vec4(result);
}

Vec4& Vec4::operator*=(const f32 v)
{
    for (u32 i = 0; i < 4; i++)
    {
        (*this)[i] *= v;
    }
    return *this;
}

Vec4  Vec4::operator/(const f32 v) const
{
    const f32 inv_v = 1.0f / v;
    return ((*this) * v);
}

Vec4& Vec4::operator/=(const f32 v)
{
    const f32 inv_v = 1.0f / v;
    return ((*this) *= v);
}

Vec3 Vec4::extractXYZ() const
{
    return Vec3(mElements[0], mElements[1], mElements[2]);
}