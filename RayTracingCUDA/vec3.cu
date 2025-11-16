#include <cassert>
#include <math.h>
#include <random>
#include "vector.h"
#include "util.h"
f32& Vec3::operator[](size_t i)
{
#ifdef DEBUG
	if (i >= 3 || i < 0)
	{
		assert(0);
	}
#endif

	return mElements[i];
}

f32 Vec3::operator[](size_t i) const
{
#ifdef DEBUG
	if (i >= 3 || i < 0)
	{
		assert(0);
	}
#endif

	return mElements[i];
}

Vec3 Vec3::operator-() const
{
	const f32 x = this->mElements[0];
	const f32 y = this->mElements[1];
	const f32 z = this->mElements[2];
	return Vec3(-x, -y, -z);
}

Vec3 Vec3::operator+(const Vec3& v) const
{
	const Vec3& u = *this;
	f32 x = u.mElements[0] + v.mElements[0];
	f32 y = u.mElements[1] + v.mElements[1];
	f32 z = u.mElements[2] + v.mElements[2];
	return Vec3(x, y, z);
}

Vec3& Vec3::operator+=(const Vec3& v)
{
	this->mElements[0] += v.mElements[0];
	this->mElements[1] += v.mElements[1];
	this->mElements[2] += v.mElements[2];
	return *this;
}

Vec3 Vec3::operator+(const f32 value) const
{
	return (*this + Vec3(value, value, value));
}

Vec3& Vec3::operator+=(const f32 value)
{
	*this += Vec3(value, value, value);
	return *this;
}

Vec3 Vec3::operator-(const Vec3& v) const
{
	const Vec3& u = *this;
	f32 x = u.mElements[0] - v.mElements[0];
	f32 y = u.mElements[1] - v.mElements[1];
	f32 z = u.mElements[2] - v.mElements[2];
	return Vec3(x, y, z);
}

Vec3& Vec3::operator-=(const Vec3& v)
{
	this->mElements[0] -= v.mElements[0];
	this->mElements[1] -= v.mElements[1];
	this->mElements[2] -= v.mElements[2];
	return *this;
}

Vec3 Vec3::operator-(const f32 value) const
{
	return (*this - Vec3(value, value, value));
}

Vec3& Vec3::operator-=(const f32 value)
{
	*this -= Vec3(value, value, value);
	return *this;
}

Vec3 Vec3::operator*(const Vec3& v) const
{
	const Vec3& u = *this;
	f32 x = u.mElements[0] * v.mElements[0];
	f32 y = u.mElements[1] * v.mElements[1];
	f32 z = u.mElements[2] * v.mElements[2];
	return Vec3(x, y, z);
}

Vec3& Vec3::operator*=(const Vec3& v)
{
	this->mElements[0] *= v.mElements[0];
	this->mElements[1] *= v.mElements[1];
	this->mElements[2] *= v.mElements[2];
	return *this;
}

Vec3 Vec3::operator*(const f32 value) const
{
	return (*this * Vec3(value, value, value));
}

Vec3& Vec3::operator*=(const f32 value)
{
	*this *= Vec3(value, value, value);
	return *this;
}

Vec3 Vec3::operator/(const f32 value) const
{
	if (value == 0.0)
	{
		assert(false);
	}
	f32 inv_value = 1.0f / value;
	return (Vec3(inv_value, inv_value, inv_value) *= *this);
}

Vec3& Vec3::operator/=(const f32 value)
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

Vec3 Vec3::normalize() const
{
	const f32 length = this->length();
	if (length == 0.0)
	{
		assert(0);
	}

	f32 x = this->mElements[0] / length;
	f32 y = this->mElements[1] / length;
	f32 z = this->mElements[2] / length;

	return Vec3(x, y, z);
}

Vec3 Vec3::normalize(const Vec3& v)
{
	return v.normalize();
}

f32 Vec3::length() const
{
	const f32 lengthSquared = this->lengthSquared();
	return sqrtf(lengthSquared);
}

f32 Vec3::length(const Vec3& v)
{
	return v.length();
}

f32 Vec3::lengthSquared() const
{
	const f32 x = this->mElements[0];
	const f32 y = this->mElements[1];
	const f32 z = this->mElements[2];
	const f32 length2 = x * x + y * y + z * z;
	return length2;
}

f32 Vec3::lengthSquared(const Vec3& v)
{
	return v.lengthSquared();
}

f32 Vec3::dot(const Vec3& v0, const Vec3& v1)
{
	const f32 v0_x = v0.x();
	const f32 v0_y = v0.y();
	const f32 v0_z = v0.z();

	const f32 v1_x = v1.x();
	const f32 v1_y = v1.y();
	const f32 v1_z = v1.z();

	return v0_x * v1_x + v0_y * v1_y + v0_z * v1_z;
}

Vec3 Vec3::cross(const Vec3& v0, const Vec3& v1)
{
	const f32 v0_x = v0.x();
	const f32 v0_y = v0.y();
	const f32 v0_z = v0.z();

	const f32 v1_x = v1.x();
	const f32 v1_y = v1.y();
	const f32 v1_z = v1.z();

	const f32 v_x = v0_y * v1_z - v0_z * v1_y;
	const f32 v_y = v0_z * v1_x - v0_x * v1_z;
	const f32 v_z = v0_x * v1_y - v0_y * v1_x;
	return Vec3(v_x, v_y, v_z);
}

Vec3 Vec3::reflect(const Vec3& v, const Vec3& n)
{
	return v - 2 * dot(v, n) * n;
}

#ifdef _DEBUG
void Vec3::debugPrint(const char* message) const
{
	printf("vector = (%f, %f, %f) : %s\n", mElements[0], mElements[1], mElements[2], message);
}
#endif

Vec3 Vec3::generateRandomUnitVector()
{
#ifdef __CUDA_ARCH__
	const f32 phi = 2 * M_PI * RandomGeneratorGPU::uniform_real();
	const f32 theta = M_PI * RandomGeneratorGPU::uniform_real();
#else
	const f32 phi = 2 * M_PI * RandomGenerator::uniform_real();
	const f32 theta = M_PI * RandomGenerator::uniform_real();
#endif
	const f32 sin0 = sin(theta);
	return Vec3(cos(phi) * sin0, sin(phi) * sin0, cos(theta));
}

__device__ __host__ Vec3 Vec3::generateRandomlyOnUnitHemiSphere(const Vec3& normal)
{
#ifdef __CUDA_ARCH__
	const f32 phi = RandomGeneratorGPU::uniform_real() * 2 * M_PI;
	const f32 z = RandomGeneratorGPU::uniform_real();
#else
	const f32 phi = RandomGenerator::uniform_real() * 2 * M_PI;
	const f32 z = RandomGenerator::uniform_real();
#endif

	const f32 cos0 = sqrtf(1 - z * z + 0.0001f);
	const f32 x = cos(phi) * cos0;
	const f32 y = sin(phi) * cos0;

	OrthonormalBasis onb(normal);

	return onb.local(x, y, z);
}

__device__ __host__ Vec3 Vec3::generateMaximumLengthVector()
{
	constexpr f32 f32Max = std::numeric_limits<f32>::max();
	return Vec3(f32Max, f32Max, f32Max);
}

__device__ __host__ Vec3 Vec3::generateMinimumLengthVector()
{
	constexpr f32 f32Max = std::numeric_limits<f32>::max();
	return Vec3(-f32Max, -f32Max, -f32Max);
}

bool Vec3::isNan() const
{
	return (isnan(mElements[0]) || isnan(mElements[1]) || isnan(mElements[2]));
}



Vec3 operator*(const f32 value, const Vec3& v)
{
	return  (v * value);
}

