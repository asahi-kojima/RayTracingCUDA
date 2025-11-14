#pragma once
#include "matrix.h"

//Matrix x vectorÇÃå`éÆÇçÃópÇµÇƒÇ¢ÇÈÅB
struct Transform
{
public:
	Transform();
	Transform(const Transform& other);
	Transform(const Vec3& position, const Vec3& scaling, const Vec3& rotationAngle = Vec3::zero());
	Transform(const Vec3& position, const f32 scaling);

	void setScaling(f32 scale_x, f32 scale_y, f32 scale_z);
	void setScaling(f32 scale);
	void setScaling(const Vec3& scale);
	void setRotation(f32 angle_x, f32 angle_y, f32 angle_z);
	void setRotation(const Vec3& angles);
	void setTranslation(f32 x, f32 y, f32 z);
	void setTranslation(const Vec3& t);

	void updateTransformMatrices();

	const Mat4& getTransformMatrix() const;
	const Mat4& getInvTransformMatrix() const;
	const Mat4& getInvTransposeTransformMatrix() const;

	const Vec3& getScaling() const;
	const Vec3& getRotation() const;
	const Vec3& getTranslation() const;

	static Transform identity();
	static Transform scaling(const Vec3& v = Vec3::one());
	static Transform rotation(const Vec3& v = Vec3::zero());
	static Transform translation(const Vec3& v = Vec3::zero());
	static Transform translation(f32 x = 0, f32 y = 0, f32 z = 0);

private:
	void calcTransformMatrix();
	void calcInverseTransformMatrix();
	void calcInverseTransposeTransformMatrix();

	bool mIsDirty;

	Vec3 mScaling;
	Vec3 mRotation;
	Vec3 mTranslation;

	Mat4 mTransformMatrix;
	Mat4 mInvTransformMatrix;
	Mat4 mInvTransposeTransformMatrix;
};