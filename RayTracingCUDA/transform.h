#pragma once
#include "matrix.h"
#include "util.h"


class Quaternion
{
public:
	Quaternion() : w(1), x(0), y(0), z(0) {}
	Quaternion(f32 w, f32 x, f32 y, f32 z) : w(w), x(x), y(y), z(z) {}
	Quaternion(f32 angle, const Vec3& axis)
	{
		if (isEqualF32(axis.lengthSquared(), 0.0f))
		{
			printf("axis vector is zero.\n");
			assert(0);
		}

		const Vec3 normalizedAxis = axis.normalize();
		f32 halfAngle = angle * 0.5f;
		f32 s = sinf(halfAngle);

		w = cosf(halfAngle);
		x = normalizedAxis.x() * s;
		y = normalizedAxis.y() * s;
		z = normalizedAxis.z() * s;
	}


	Quaternion operator*(const Quaternion& q) const
	{
		return Quaternion(
			w * q.w - x * q.x - y * q.y - z * q.z,
			w * q.x + x * q.w + y * q.z - z * q.y,
			w * q.y - x * q.z + y * q.w + z * q.x,
			w * q.z + x * q.y - y * q.x + z * q.w
		);
	}

	Quaternion conjugate() const
	{
		return Quaternion(w, -x, -y, -z);
	}

	f32 norm() const
	{
		return std::sqrt(w * w + x * x + y * y + z * z);
	}

	Quaternion normalized() const
	{
		f32 n = norm();
		if (n == 0.0f) return Quaternion();
		return Quaternion(w / n, x / n, y / n, z / n);
	}

	Vec3 rotate(const Vec3& v) const
	{
		Quaternion qv(0, v.x(), v.y(), v.z());
		Quaternion qr = (*this) * qv * this->conjugate();
		return Vec3(qr.x, qr.y, qr.z);
	}

	Mat4 toRotationMatrix() const
	{
		f32 xx = x * x, yy = y * y, zz = z * z;
		f32 xy = x * y, xz = x * z, yz = y * z;
		f32 wx = w * x, wy = w * y, wz = w * z;

		return Mat4(
			1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0,
			2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0,
			2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0,
			0, 0, 0, 1
		);
	}

	// 2つの方向ベクトル v から w への回転を表すクオータニオンを生成する
	static Quaternion fromToRotation(const Vec3& from, const Vec3& to)
	{
		// 1. 正規化
		Vec3 v = from.normalize();
		Vec3 w = to.normalize();

		// 内積（cosθ）を計算
		f32 dot = Vec3::dot(v, w); // ※Vec3にdotメソッドがあると仮定

		// 2. ほぼ同じ向きを向いている場合（回転の必要なし）
		if (dot > 0.99999f)
		{
			return Quaternion(); // 単位クオータニオン
		}

		// 4. 完全に真逆（180度）を向いている場合の例外処理
		if (dot < -0.99999f)
		{
			// v と直交する適当な軸を探す
			Vec3 axis = Vec3::cross(Vec3(1.0f, 0.0f, 0.0f), v); // ※Vec3にcrossメソッドがあると仮定
			if (axis.lengthSquared() < 0.0001f)
			{
				// もし (1,0,0) と平行だったら (0,1,0) を使う
				axis = Vec3::cross(Vec3(0.0f, 1.0f, 0.0f),v);
			}
			return Quaternion(M_PI, axis.normalize()); // 180度（πラジアン）回転
		}

		// 3. 通常の回転（外積から軸を求め、内積から角度を求める）
		Vec3 axis = Vec3::cross(v, w);
		f32 angle = std::acos(dot);

		// 既存の「軸と角度」のコンストラクタに渡す
		return Quaternion(angle, axis);
	}

private:
	f32 w;
	f32 x;
	f32 y;
	f32 z;
};




struct Transform
{
public:
	Transform();
	Transform(const Transform& other);
	Transform(const Vec3& position, const Vec3& scaling, const Quaternion& rotation = Quaternion());
	Transform(const Vec3& position, const f32 scaling, const Quaternion& rotation = Quaternion());

	void setScaling(f32 scale_x, f32 scale_y, f32 scale_z);
	void setScaling(f32 scale);
	void setScaling(const Vec3& scale);
	void setRotation(f32 angle_x, f32 angle_y, f32 angle_z);
	void setRotation(const f32 angle, const Vec3& axis);
	void setTranslation(f32 x, f32 y, f32 z);
	void setTranslation(const Vec3& t);

	const Vec3& getScaling() const;
	const Quaternion& getRotation() const;
	const Vec3& getTranslation() const;

	void updateTransformMatrices();

	const Mat4& getTransformMatrix() const;
	const Mat4& getInvTransformMatrix() const;
	const Mat4& getInvTransposeTransformMatrix() const;


	static Transform identity();
	static Transform scaling(const Vec3& v = Vec3::one());
	static Transform rotation(const f32 angle = 0.0f, const Vec3& axis = Vec3::zero());
	static Transform translation(const Vec3& v = Vec3::zero());
	static Transform translation(f32 x = 0, f32 y = 0, f32 z = 0);

private:
	void calcTransformMatrix();
	void calcInverseTransformMatrix();
	void calcInverseTransposeTransformMatrix();

	bool mIsDirty;

	Vec3 mScaling;
	Quaternion mRotation;
	Vec3 mTranslation;

	Mat4 mTransformMatrix;
	Mat4 mInvTransformMatrix;
	Mat4 mInvTransposeTransformMatrix;
};