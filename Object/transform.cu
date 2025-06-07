#include "transform.h"



Transform::Transform()
: mScaling{1.0f, 1.0f, 1.0f}
, mRotation{0.0f, 0.0f, 0.0f}
, mTranslation{0, 0, 0} 
, mTransformMatrix{}
, mInvTransformMatrix{}
{}





void Transform::setScaling(f32 scale_x, f32 scale_y, f32 scale_z)
{
#ifdef DEBUG
    if (scale_x == 0.0f || scale_y == 0.0f || scale_z == 0.0f)
    {
        printf("Error : scale value is 0!\n");
    }
#endif
    mScaling[0] = scale_x;
    mScaling[1] = scale_y;
    mScaling[2] = scale_z;
    mIsDirty = true;
}

void Transform::setScaling(f32 scale)
{
    setScaling(scale, scale, scale);
}

void Transform::setRotationAngle(f32 angle_x, f32 angle_y, f32 angle_z)
{
    mRotation[0] = angle_x;
    mRotation[1] = angle_y;
    mRotation[2] = angle_z;
    mIsDirty = true;
}


void Transform::setTranslation(const Vec3& t)
{
    mTranslation = t;
    mIsDirty = true;
}

const Mat4& Transform::getTransformMatrix()
{
    if (mIsDirty)
    {
        calcTransformMatrix();
        calcInverseTransformMatrix();
        calcInverseTransposeTransformMatrix();
        mIsDirty = false;
    }

    return mTransformMatrix;
}


const Mat4& Transform::getInvTransformMatrix()
{
    if (mIsDirty)
    {
        calcTransformMatrix();
        calcInverseTransformMatrix();
        calcInverseTransposeTransformMatrix();
        mIsDirty = false;
    }

    return mInvTransformMatrix;
}

const Mat4& Transform::getInvTransposeTransformMatrix()
{
    if (mIsDirty)
    {
        calcTransformMatrix();
        calcInverseTransformMatrix();
        calcInverseTransposeTransformMatrix();
        mIsDirty = false;
    }

    return mInvTransposeTransformMatrix;
}


const Vec3& Transform::getTranslation() const
{
    return mTranslation;
}


Transform Transform::translation(const Vec3& v)
{
    Transform transform{};
    transform.setTranslation(v);
    return transform;
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
    const Mat4 inv_R = Mat4::generateRotation(-mRotation[0], -mRotation[1], -mRotation[2]);
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