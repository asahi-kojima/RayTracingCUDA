#include <cassert>
#include <math.h>
#include <random>
#include "vector.h"
#include "util.h"

f32& vec3::operator[](size_t i)
{
	if (i >= 3 || i < 0)
	{
		assert(0);
	}
	return mElements[i];
}

f32 vec3::operator[](size_t i) const
{
	if (i >= 3 || i < 0)
	{
		assert(0);
	}
	return mElements[i];
}

vec3 vec3::operator-() const
{
	const f32 x = this->mElements[0];
	const f32 y = this->mElements[1];
	const f32 z = this->mElements[2];
	return vec3(-x, -y, -z);
}

vec3 vec3::operator+(const vec3& v) const
{
	const vec3& u = *this;
	f32 x = u.mElements[0] + v.mElements[0];
	f32 y = u.mElements[1] + v.mElements[1];
	f32 z = u.mElements[2] + v.mElements[2];
	return vec3(x, y, z);
}

vec3& vec3::operator+=(const vec3& v)
{
	this->mElements[0] += v.mElements[0];
	this->mElements[1] += v.mElements[1];
	this->mElements[2] += v.mElements[2];
	return *this;
}

vec3 vec3::operator+(const f32 value) const
{
	return (*this + vec3(value, value, value));
}

vec3& vec3::operator+=(const f32 value)
{
	*this += vec3(value, value, value);
	return *this;
}

vec3 vec3::operator-(const vec3& v) const
{
	const vec3& u = *this;
	f32 x = u.mElements[0] - v.mElements[0];
	f32 y = u.mElements[1] - v.mElements[1];
	f32 z = u.mElements[2] - v.mElements[2];
	return vec3(x, y, z);
}

vec3& vec3::operator-=(const vec3& v)
{
	this->mElements[0] -= v.mElements[0];
	this->mElements[1] -= v.mElements[1];
	this->mElements[2] -= v.mElements[2];
	return *this;
}

vec3 vec3::operator-(const f32 value) const
{
	return (*this - vec3(value, value, value));
}

vec3& vec3::operator-=(const f32 value)
{
	*this -= vec3(value, value, value);
	return *this;
}

vec3 vec3::operator*(const vec3& v) const
{
	const vec3& u = *this;
	f32 x = u.mElements[0] * v.mElements[0];
	f32 y = u.mElements[1] * v.mElements[1];
	f32 z = u.mElements[2] * v.mElements[2];
	return vec3(x, y, z);
}

vec3& vec3::operator*=(const vec3& v)
{
	this->mElements[0] *= v.mElements[0];
	this->mElements[1] *= v.mElements[1];
	this->mElements[2] *= v.mElements[2];
	return *this;
}

vec3 vec3::operator*(const f32 value) const
{
	return (*this * vec3(value, value, value));
}

vec3& vec3::operator*=(const f32 value)
{
	*this *= vec3(value, value, value);
	return *this;
}

vec3 vec3::operator/(const f32 value) const
{
	if (value == 0.0)
	{
		assert(false);
	}
	f32 inv_value = 1.0f / value;
	return (vec3(inv_value, inv_value, inv_value) *= *this);
}

vec3& vec3::operator/=(const f32 value)
{
	if (value == 0.0)
	{
		assert(false);
	}
	f32 inv_value = 1.0f / value;
	this->mElements[0] *= inv_value;
	this->mElements[1] *= inv_value;
	this->mElements[2] *= inv_value;
	return *this;
}

vec3 vec3::normalize()
{
	const f32 length = this->length();
	if (length == 0.0)
	{
		assert(0);
	}

	f32 x = this->mElements[0] / length;
	f32 y = this->mElements[1] / length;
	f32 z = this->mElements[2] / length;

	return vec3(x, y, z);
}

f32 vec3::length() const
{
	const f32 lengthSquared = this->lengthSquared();
	return sqrtf(lengthSquared);
}

f32 vec3::length(const vec3& v)
{
	return v.length();
}

f32 vec3::lengthSquared() const
{
	const f32 x = this->mElements[0];
	const f32 y = this->mElements[1];
	const f32 z = this->mElements[2];
	const f32 length2 = x * x + y * y + z * z;
	return length2;
}

f32 vec3::lengthSquared(const vec3& v)
{
	return v.lengthSquared();
}

vec3 vec3::cross(const vec3& v0, const vec3& v1)
{
	const f32 v0_x = v0.getX();
	const f32 v0_y = v0.getY();
	const f32 v0_z = v0.getZ();

	const f32 v1_x = v1.getX();
	const f32 v1_y = v1.getY();
	const f32 v1_z = v1.getZ();

	const f32 v_x = v0_y * v1_z - v0_z * v1_y;
	const f32 v_y = v0_z * v1_x - v0_x * v1_z;
	const f32 v_z = v0_x * v1_y - v0_y * v1_x;
	return vec3(v_x, v_y, v_z);
}

void vec3::print() const
{
	printf("vector = (%f, %f, %f)\n", mElements[0], mElements[1], mElements[2]);
}

//=================================================
// ��������N���X�O�֐�
//=================================================

vec3 operator*(const f32 value, const vec3& v)
{
	return  (v * value);
}


vec3 normalize(const vec3& v)
{
	vec3 normalized_v = v;

	return normalized_v.normalize();
}

vec3 cross(const vec3& v0, const vec3& v1)
{
	const f32 v0_x = v0.getX();
	const f32 v0_y = v0.getY();
	const f32 v0_z = v0.getZ();

	const f32 v1_x = v1.getX();
	const f32 v1_y = v1.getY();
	const f32 v1_z = v1.getZ();

	const f32 v_x = v0_y * v1_z - v0_z * v1_y;
	const f32 v_y = v0_z * v1_x - v0_x * v1_z;
	const f32 v_z = v0_x * v1_y - v0_y * v1_x;
	return vec3(v_x, v_y, v_z);
}

f32 dot(const vec3& v0, const vec3& v1)
{
	const f32 v0_x = v0.getX();
	const f32 v0_y = v0.getY();
	const f32 v0_z = v0.getZ();

	const f32 v1_x = v1.getX();
	const f32 v1_y = v1.getY();
	const f32 v1_z = v1.getZ();

	return v0_x * v1_x + v0_y * v1_y + v0_z * v1_z;
}

vec3 reflect(const vec3& v, const vec3& n)
{
	return v - 2 * dot(v, n) * n;
}




vec3 random_in_unit_sphere()
{
	vec3 p;
	do
	{
		p = vec3(RandomGeneratorGPU::signed_uniform_real(), RandomGeneratorGPU::signed_uniform_real(), RandomGeneratorGPU::signed_uniform_real());
	} while (p.lengthSquared() >= 1.0f);
	return p;

	return vec3::zero();
}