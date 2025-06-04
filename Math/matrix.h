#pragma once
#include "common.h"
#include "vector.h"

class Mat3
{
public:
    __device__ __host__ Mat3(const Vec3& v0, const Vec3& v1, const Vec3& v2)
    : mRowVectorList{v0, v1, v2}
    {}

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


Vec3 operator*(const Mat3& M, const Vec3& v)
{
    Vec3 Mv;
    for (u32 i = 0; i < 3; i++)
    {
        f32 tmp = 0.0f;
        Mv[i] = M(i, 0) * v[0] + M(i, 1) * v[1] + M(i, 2) * v[2];
    }

    return Mv;
}



class Mat4
{
public:
    __device__ __host__ Mat4()
    : mElementList{
        Vec4(1.0f, 0.0f, 0.0f, 0.0f), 
        Vec4(0.0f, 1.0f, 0.0f, 0.0f), 
        Vec4(0.0f, 0.0f, 1.0f, 0.0f), 
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)}
    {}

    __device__ __host__ Mat4(const Vec4& v0, const Vec4& v1, const Vec4& v2,const Vec4& v3)
    : mElementList{v0, v1, v2, v3}
    {}

    __device__ __host__ Mat4(
        f32 v00, f32 v01, f32 v02, f32 v03,
        f32 v10, f32 v11, f32 v12, f32 v13,
        f32 v20, f32 v21, f32 v22, f32 v23,
        f32 v30, f32 v31, f32 v32, f32 v33)
    : mElementList{
        Vec4(v00, v01, v02, v03), 
        Vec4(v10, v11, v12, v13), 
        Vec4(v20, v21, v22, v23), 
        Vec4(v30, v31, v32, v33)}
    {}

    __device__ __host__ Mat4(const Mat4& m)
    : mElementList{m.mElementList.asVector[0], mElementList.asVector[1], mElementList.asVector[2], mElementList.asVector[3]}
    {}


    __device__ __host__ f32 operator()(size_t i, size_t j) const;
    __device__ __host__ f32& operator()(size_t i, size_t j);

    //------------------------------------------------------------
    // operator overload
    //------------------------------------------------------------
    __device__ __host__ Mat4 operator*(const Mat4& other);

    //------------------------------------------------------------
    // 平行移動行列を作成する
    //------------------------------------------------------------
    __device__ __host__ static Mat4 generateTransform(f32 x, f32 y, f32 z);
    __device__ __host__ static Mat4 generateTransform(const Vec3& v);

    //------------------------------------------------------------
    // 平行移動行列を作成する
    //------------------------------------------------------------
    __device__ __host__ static Mat4 generateRotation(f32 angleX, f32 angleY, f32 angleZ);
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

Vec4 operator*(const Mat4& M, const Vec4& v);