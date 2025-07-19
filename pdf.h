#pragma once
#include "util.h"
#include "Math/vector.h"

class PDF
{
public:
    __device__ virtual ~PDF() = default;

    __device__ virtual f32 value(const Vec3& direction) const = 0;
    __device__ virtual Vec3 generateRandomDirection() const = 0; 
};


class CosinePdf : public PDF
{
public:
    __device__ CosinePdf(const Vec3& baseDirection) : mOnb(baseDirection){}
    __device__ ~CosinePdf() = default;

    __device__ virtual f32 value(const Vec3& direction) const override;
    __device__ virtual Vec3 generateRandomDirection() const override;
    
private:
    ONB mOnb;
};