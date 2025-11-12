#pragma once
#include <stdio.h>
#include "common.h"
#include "vector.h"

class Mat3
{
public:
    __device__ __host__ Mat3(const Vec3& v0, const Vec3& v1, const Vec3& v2)
        : mRowVectorList{ v0, v1, v2 }
    {
    }

    __device__ __host__ f32 operator()(u32 i, u32 j) const
    {
#ifdef DEBUG
        if (!(i < 3 && j < 3))
        {
            printf("out of Range\n");
        }
#endif
        return mRowVectorList[i][j];
    }

    __device__ __host__ f32& operator()(u32 i, u32 j)
    {
#ifdef DEBUG
        if (!(i < 3 && j < 3))
        {
            printf("out of Range\n");
        }
#endif
        return mRowVectorList[i][j];
    }

private:
    Vec3 mRowVectorList[3];
};





#define PRINT_MATRIX(R)\
printf("""\
=================================\n\
%5f,%5f,%5f,%5f\n\
%5f,%5f,%5f,%5f\n\
%5f,%5f,%5f,%5f\n\
%5f,%5f,%5f,%5f\n\
=================================\n""", R(0,0), R(0,1), R(0,2), R(0,3), R(1,0), R(1,1), R(1,2), R(1,3), R(2,0), R(2,1), R(2,2), R(2,3), R(3,0), R(3,1), R(3,2), R(3,3));

class Mat4
{
public:
    __device__ __host__ Mat4()
        : mElementList{
            Vec4(1.0f, 0.0f, 0.0f, 0.0f),
            Vec4(0.0f, 1.0f, 0.0f, 0.0f),
            Vec4(0.0f, 0.0f, 1.0f, 0.0f),
            Vec4(0.0f, 0.0f, 0.0f, 1.0f) }
    {
    }

    __device__ __host__ Mat4(const Vec4& v0, const Vec4& v1, const Vec4& v2, const Vec4& v3)
        : mElementList{ v0, v1, v2, v3 }
    {
    }

    __device__ __host__ Mat4(
        f32 v00, f32 v01, f32 v02, f32 v03,
        f32 v10, f32 v11, f32 v12, f32 v13,
        f32 v20, f32 v21, f32 v22, f32 v23,
        f32 v30, f32 v31, f32 v32, f32 v33)
        : mElementList{
            Vec4(v00, v01, v02, v03),
            Vec4(v10, v11, v12, v13),
            Vec4(v20, v21, v22, v23),
            Vec4(v30, v31, v32, v33) }
    {
    }

    __device__ __host__ Mat4(const Mat4& m)
        : Mat4{ m.mElementList.asVector[0], m.mElementList.asVector[1], m.mElementList.asVector[2], m.mElementList.asVector[3] }
    {
    }


    __device__ __host__ f32 operator()(size_t i, size_t j) const;
    __device__ __host__ f32& operator()(size_t i, size_t j);

    __device__ __host__ Mat4 transpose();

    //------------------------------------------------------------
    // operator overload
    //------------------------------------------------------------
    __device__ __host__ Mat4 operator*(const Mat4& other) const;


    //------------------------------------------------------------
    // 平行移動行列を作成する
    //------------------------------------------------------------
    __device__ __host__ static Mat4 generateTranslation(f32 x, f32 y, f32 z);
    __device__ __host__ static Mat4 generateTranslation(const Vec3& v);

    //------------------------------------------------------------
    // 回転行列を作成する
    //------------------------------------------------------------
    __device__ __host__ static Mat4 generateRotation(f32 angleX, f32 angleY, f32 angleZ);
    __device__ __host__ static Mat4 generateInverseRotation(f32 angleX, f32 angleY, f32 angleZ);
    __device__ __host__ static Mat4 generateRotation(const Vec3& rotationUnitVector, f32 angle);
    __device__ __host__ static Mat4 generateRotation(const Vec3& rotationVector);

    //------------------------------------------------------------
    // スケーリング行列を作成する
    //------------------------------------------------------------
    __device__ __host__ static Mat4 generateScale(f32 sx, f32 sy, f32 sz);

private:
    union
    {
        Vec4 asVector[4];
        f32 asF32[16];
    } mElementList;
};

__device__ __host__ Vec4 operator*(const Mat4& M, const Vec4& v);