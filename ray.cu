#include "ray.h"

const Vec3& Ray::origin() const 
{
    return mOrigin;
}

Vec3& Ray::origin()
{
    return mOrigin;
}

const Vec3& Ray::direction() const
{
    return mDirection;
}

Vec3& Ray::direction()
{
    return mDirection;
}

Vec3 Ray::pointAt(const f32 t) const
{
    return mOrigin + mDirection * t;
}


Ray Ray::transformWith(const Mat4& mat) const 
{
    Vec4 origin(mOrigin, 1.0f);
    Vec4 direction(mDirection, 0.0f);

    Vec3 transformedOrigin = (mat * origin).extractXYZ();
    Vec3 transformedDirection = (mat * direction).extractXYZ();


    return Ray(transformedOrigin, transformedDirection);
}


void Ray::print_debug() const
{
    mOrigin.print_debug("Origin");
    mDirection.print_debug("Direction");
}