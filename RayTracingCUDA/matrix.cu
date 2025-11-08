#include "matrix.h"

Vec3 operator*(const Mat3& M, const Vec3& v)
{
    Vec3 Mv;
    for (u32 i = 0; i < 3; i++)
    {
        Mv[i] = M(i, 0) * v[0] + M(i, 1) * v[1] + M(i, 2) * v[2];
    }

    return Mv;
}


f32 Mat4::operator()(size_t i, size_t j) const
{
#ifdef DEBUG
    if (!(i >= 0 && i < 4))
    {
        printf("out of Range\n");
    }
#endif
    return mElementList.asVector[i][j];
}

f32& Mat4::operator()(size_t i, size_t j)
{
#ifdef DEBUG
    if (!(i >= 0 && i < 4))
    {
        printf("out of Range\n");
    }
#endif
    return mElementList.asVector[i][j];
}


Mat4 Mat4::transpose()
{
    return Mat4(
        mElementList.asVector[0][0], mElementList.asVector[1][0], mElementList.asVector[2][0], mElementList.asVector[3][0],
        mElementList.asVector[0][1], mElementList.asVector[1][1], mElementList.asVector[2][1], mElementList.asVector[3][1],
        mElementList.asVector[0][2], mElementList.asVector[1][2], mElementList.asVector[2][2], mElementList.asVector[3][2],
        mElementList.asVector[0][3], mElementList.asVector[1][3], mElementList.asVector[2][3], mElementList.asVector[3][3]
    );
}


Mat4 Mat4::operator*(const Mat4& other) const
{
    Mat4 m;
    for (size_t col = 0; col < 4; col++)
    {
        const f32 v0 = other(0, col);
        const f32 v1 = other(1, col);
        const f32 v2 = other(2, col);
        const f32 v3 = other(3, col);

        m(0, col) = (*this)(0, 0) * v0 + (*this)(0, 1) * v1 + (*this)(0, 2) * v2 + (*this)(0, 3) * v3;
        m(1, col) = (*this)(1, 0) * v0 + (*this)(1, 1) * v1 + (*this)(1, 2) * v2 + (*this)(1, 3) * v3;
        m(2, col) = (*this)(2, 0) * v0 + (*this)(2, 1) * v1 + (*this)(2, 2) * v2 + (*this)(2, 3) * v3;
        m(3, col) = (*this)(3, 0) * v0 + (*this)(3, 1) * v1 + (*this)(3, 2) * v2 + (*this)(3, 3) * v3;
    }

    return m;
}

Mat4 Mat4::generateTranslation(f32 x, f32 y, f32 z)
{
    return Mat4(
        Vec4(1, 0, 0, x),
        Vec4(0, 1, 0, y),
        Vec4(0, 0, 1, z),
        Vec4(0, 0, 0, 1));
}

Mat4 Mat4::generateTranslation(const Vec3& v)
{
    return Mat4(
        Vec4(1, 0, 0, v.x()),
        Vec4(0, 1, 0, v.y()),
        Vec4(0, 0, 1, v.z()),
        Vec4(0, 0, 0, 1));
}

Mat4 Mat4::generateRotation(f32 angleX, f32 angleY, f32 angleZ)
{
    const f32 cosx = cos(angleX);
    const f32 sinx = sin(angleX);
    const f32 cosy = cos(angleY);
    const f32 siny = sin(angleY);
    const f32 cosz = cos(angleZ);
    const f32 sinz = sin(angleZ);

    return Mat4(
        cosy * cosz, sinx * siny * cosz - cosx * sinz, cosx * siny * cosz + sinx * sinz, 0,
        cosy * sinz, sinx * siny * sinz + cosx * cosz, cosx * siny * sinz - sinx * cosz, 0,
        -siny, sinx * cosy, cosx * cosy, 0,
        0, 0, 0, 1
    );
}

Mat4 Mat4::generateInverseRotation(f32 angleX, f32 angleY, f32 angleZ)
{
    const f32 cosx = cos(angleX);
    const f32 sinx = sin(angleX);
    const f32 cosy = cos(angleY);
    const f32 siny = sin(angleY);
    const f32 cosz = cos(angleZ);
    const f32 sinz = sin(angleZ);

    return Mat4(
        cosy * cosz, -cosy * sinz, siny, 0,
        sinx * siny * cosz + cosx * sinz, -sinx * siny * sinz + cosx * cosz, -sinx * cosy, 0,
        -cosx * siny * cosz + sinx * sinz, cosx * siny * sinz + sinx * cosz, cosx * cosy, 0,
        0, 0, 0, 1
    );
}



Mat4 Mat4::generateRotation(const Vec3& rotationUnitVector, f32 angle)
{
    const Vec3& n = rotationUnitVector;

    //共通の値などをキャッシュ
    const f32 cos0 = cos(angle);
    const f32 sin0 = sin(angle);
    const f32 complement_cos0 = 1 - cos0;

    const f32 nx = n.x();
    const f32 ny = n.y();
    const f32 nz = n.z();
    const f32 nxnx = nx * nx;
    const f32 nyny = ny * ny;
    const f32 nznz = nz * nz;
    const f32 nxny = nx * ny;
    const f32 nynz = ny * nz;
    const f32 nznx = nz * nx;

    Vec4 row0(
        nxnx * complement_cos0 + cos0,
        nxny * complement_cos0 - nz * sin0,
        nznx * complement_cos0 + ny * sin0,
        0);
    Vec4 row1(
        nxny * complement_cos0 + nz * sin0,
        nyny * complement_cos0 + cos0,
        nynz * complement_cos0 - nx * sin0,
        0);
    Vec4 row2(
        nznx * complement_cos0 - ny * sin0,
        nynz * complement_cos0 + nx * sin0,
        nz * nz * complement_cos0 + cos0,
        0);
    Vec4 row3(0, 0, 0, 1);

    return Mat4(row0, row1, row2, row3);
}

Mat4 Mat4::generateRotation(const Vec3& rotationVector)
{
    return generateRotation(rotationVector.normalize(), rotationVector.length());
}

Mat4 Mat4::generateScale(f32 sx, f32 sy, f32 sz)
{
    return Mat4(
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, sz, 0,
        0, 0, 0, 1);
}





Vec4 operator*(const Mat4& M, const Vec4& v)
{
    Vec4 Mv;
    for (u32 i = 0; i < 4; i++)
    {
        Mv[i] = M(i, 0) * v[0] + M(i, 1) * v[1] + M(i, 2) * v[2] + M(i, 3) * v[3];
    }

    return Mv;
}