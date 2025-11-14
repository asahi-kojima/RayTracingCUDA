#include "transform.h"



Transform::Transform()
    : mScaling{ 1.0f, 1.0f, 1.0f }
    , mRotation{ 0.0f, 0.0f, 0.0f }
    , mTranslation{ 0, 0, 0 }
    , mTransformMatrix{}
    , mInvTransformMatrix{}
    , mInvTransposeTransformMatrix{}
    , mIsDirty(true)
{
}

Transform::Transform(const Transform& other)
    : mScaling{ other.mScaling }
    , mRotation{ other.mRotation }
    , mTranslation{ other.mTranslation}
    , mTransformMatrix{ other.mTransformMatrix}
    , mInvTransformMatrix{ other.mInvTransformMatrix}
    , mInvTransposeTransformMatrix{ other.mInvTransposeTransformMatrix}
    , mIsDirty(other.mIsDirty)
{
}

Transform::Transform(const Vec3& position, const Vec3& scaling, const Vec3& rotationAngle)
	: mScaling{ scaling }
	, mRotation{ rotationAngle }
	, mTranslation{ position }
	, mTransformMatrix{}
	, mInvTransformMatrix{}
	, mInvTransposeTransformMatrix{}
	, mIsDirty(true)
{
}

Transform::Transform(const Vec3& position, const f32 scaling)
	: mScaling{ scaling, scaling, scaling }
	, mRotation{ 0.0f, 0.0f, 0.0f }
	, mTranslation{ position }
	, mTransformMatrix{}
	, mInvTransformMatrix{}
	, mInvTransposeTransformMatrix{}
	, mIsDirty(true)
{
}


void Transform::setScaling(const Vec3& scale)
{
#ifdef _DEBUG
    const f32 ep = 1e-8;
    if (abs(scale[0]) < ep || abs(scale[0]) < ep || abs(scale[0]) < ep)
    {
        printf("Error : scale value is 0!\n");
    }
#endif
    mScaling = scale;
    mIsDirty = true;
}

void Transform::setScaling(f32 scale_x, f32 scale_y, f32 scale_z)
{
    setScaling(Vec3(scale_x, scale_y, scale_z));
}

void Transform::setScaling(f32 scale)
{
    setScaling(scale, scale, scale);
}

void Transform::setRotation(f32 angle_x, f32 angle_y, f32 angle_z)
{
    mRotation[0] = angle_x;
    mRotation[1] = angle_y;
    mRotation[2] = angle_z;
    mIsDirty = true;
}

void Transform::setRotation(const Vec3& angles)
{
    setRotation(angles[0], angles[1], angles[2]);
}

void Transform::setTranslation(f32 x, f32 y, f32 z)
{
    mTranslation = Vec3(x, y, z);
    mIsDirty = true;
}

void Transform::setTranslation(const Vec3& t)
{
    setTranslation(t[0], t[1], t[2]);
}

void Transform::updateTransformMatrices()
{
    if (mIsDirty)
    {
        calcTransformMatrix();
        calcInverseTransformMatrix();
        calcInverseTransposeTransformMatrix();
        mIsDirty = false;
    }
}

const Mat4& Transform::getTransformMatrix() const
{
    return mTransformMatrix;
}


const Mat4& Transform::getInvTransformMatrix() const
{
    return mInvTransformMatrix;
}

const Mat4& Transform::getInvTransposeTransformMatrix() const
{
    return mInvTransposeTransformMatrix;
}


const Vec3& Transform::getScaling() const
{
    return mScaling;
}

const Vec3& Transform::getRotation() const
{
    return mRotation;
}

const Vec3& Transform::getTranslation() const
{
    return mTranslation;
}


Transform Transform::identity()
{
    return Transform{};
}

Transform Transform::scaling(const Vec3& v)
{
    Transform transform{};
    transform.setScaling(v);
    return transform;
}

Transform Transform::rotation(const Vec3& v)
{
    Transform transform{};
    transform.setRotation(v);
    return transform;
}

Transform Transform::translation(const Vec3& v)
{
    Transform transform{};
    transform.setTranslation(v);
    return transform;
}

Transform Transform::translation(f32 x, f32 y, f32 z)
{
    return Transform::translation(Vec3(x, y, z));
}



void Transform::calcTransformMatrix()
{
    const Mat4 S = Mat4::generateScale(mScaling[0], mScaling[1], mScaling[2]);
    const Mat4 R = Mat4::generateRotation(mRotation[0], mRotation[1], mRotation[2]);
    const Mat4 T = Mat4::generateTranslation(mTranslation);

    mTransformMatrix = T * R * S;
}

void Transform::calcInverseTransformMatrix()
{
    const Mat4 inv_T = Mat4::generateTranslation(-mTranslation);
    const Mat4 inv_R = Mat4::generateInverseRotation(-mRotation[0], -mRotation[1], -mRotation[2]);
    const Mat4 inv_S = Mat4::generateScale(1.0f / mScaling[0], 1.0f / mScaling[1], 1.0f / mScaling[2]);

    mInvTransformMatrix = inv_S * inv_R * inv_T;
}

void Transform::calcInverseTransposeTransformMatrix()
{
    const Mat4 invTranspose_S = Mat4::generateScale(1.0f / mScaling[0], 1.0f / mScaling[1], 1.0f / mScaling[2]);
    const Mat4 invTranspose_R = Mat4::generateRotation(mRotation[0], mRotation[1], mRotation[2]);
    const Mat4 invTranspose_T = Mat4::generateTranslation(-mTranslation).transpose();

    mInvTransposeTransformMatrix = invTranspose_T * invTranspose_R * invTranspose_S;
}