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

class Object;
class ObjectPdf : public PDF
{
public:
    __device__ ObjectPdf(Object* object, const Vec3& origin) : mObject(object), mOrigin(origin){}
    __device__ ~ObjectPdf() = default;

    __device__ virtual f32 value(const Vec3& direction) const override;
    __device__ virtual Vec3 generateRandomDirection() const override;
    
private:
    Object* mObject;
    const Vec3& mOrigin;
};

class MixturePdf : public PDF
{
public:
    __device__ MixturePdf(const PDF& pdf0, const PDF& pdf1, f32 alpha = 0.5f) : mPdf0(pdf0), mPdf1(pdf1), mAlpha(alpha) {}
    __device__ ~MixturePdf() = default;

    __device__ virtual f32 value(const Vec3& direction) const override;
    __device__ virtual Vec3 generateRandomDirection() const override;
    
private:
    const PDF& mPdf0;
    const PDF& mPdf1;
    const f32 mAlpha;
};