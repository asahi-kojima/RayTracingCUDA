#include <stdio.h>
#include "color.h"


Color::Color(const f32 r, const f32 g, const f32 b, const f32 alpha)
	: mRGB(r, g, b), mAlpha(alpha)
{

}

Color::Color(const Vec3& rbg, const f32 alpha)
	: mRGB(rbg), mAlpha(alpha)
{
}



Color::Color(const u32 hexadecimal)
	: mRGB{ static_cast<f32>((hexadecimal & 0xFF0000) >> 16) / 255.0f,  static_cast<f32>((hexadecimal & 0x00FF00) >> 8) / 255.0f, static_cast<f32>((hexadecimal & 0x0000FF) >> 0) / 255.0f }, mAlpha(1.0f)
{
	if (hexadecimal > 0xFFFFFF)
	{
		assert(0);
	}
}

f32 Color::operator[](size_t i)const
{
#ifdef _DEBUG
	if (!(i >= 0 && i < 4))
	{
		assert(0);
	}
#endif
	if (i == 3)
	{
		return mAlpha;
	}

	return mRGB[i];
}

f32& Color::operator[](size_t i)
{
#ifdef _DEBUG
	if (!(i >= 0 && i < 4))
	{
		assert(0);
	}
#endif
	if (i == 3)
	{
		return mAlpha;
	}

	return mRGB[i];
}

__device__ __host__ f32& Color::r()
{
	return mRGB[0];
}

__device__ __host__ f32& Color::g()
{
	return mRGB[1];
}

__device__ __host__ f32& Color::b()
{
	return mRGB[2];
}

__device__ __host__ f32 Color::r() const
{
	return mRGB[0];
}

__device__ __host__ f32 Color::g() const
{
	return mRGB[1];
}

__device__ __host__ f32 Color::b() const
{
	return mRGB[2];
}


Color Color::operator+(const Color& rhs) const
{
	return Color(this->mRGB + rhs.mRGB, this->mAlpha + rhs.mAlpha);
}

Color Color::operator*(const Color& rhs) const
{
	return Color(this->mRGB * rhs.mRGB, this->mAlpha * rhs.mAlpha);
}

Color Color::operator*(const f32 value) const
{
	return Color(this->mRGB * value, this->mAlpha * value);
}

Color& Color::operator+=(const Color& rhs)
{
	this->mRGB += rhs.mRGB;
	this->mAlpha += rhs.mAlpha;
	return *this;
}

Color& Color::operator+=(const f32 rhs)
{
	this->mRGB += rhs;
	this->mAlpha += rhs;
	return *this;
}

Color& Color::operator*=(const Color& rhs)
{
	this->mRGB *= rhs.mRGB;
	this->mAlpha *= rhs.mAlpha;
	return *this;
}

Color& Color::operator*=(const f32 value)
{
	this->mRGB *= value;
	this->mAlpha *= value;
	return *this;
}

Color& Color::operator/=(const f32 value)
{
	if (value == 0.0f)
	{
		assert(0);
	}
	return ((*this) *= (1.0f / value));
}

Vec3 Color::getRGB() const
{
	return mRGB;
}


__device__ void Color::printColor() const
{
	printf("color = (%f, %f, %f)\n", mRGB[0], mRGB[1], mRGB[2]);
}


__device__ void Color::clamp()
{
	for (u32 i = 0; i < 3; i++)
	{
		const f32 tmp = mRGB[i];
		mRGB[i] = (tmp > 1.0f ? 1.0f : tmp);
	}
}

bool Color::isNan()
{
	return mRGB.isNan();
}




const Color Color::White(0xFFFFFF);
const Color Color::Black(0x000000);
const Color Color::Gray(0x808080);
const Color Color::Red(0xFF0000);
const Color Color::Lime(0x00FF00);
const Color Color::Green(0x008000);
const Color Color::Blue(0x0000FF);
const Color Color::Yellow(0xFFFF00);
const Color Color::Magenta(0xFF00FF);
const Color Color::Orange(0xFFA500);
const Color Color::Gold(0xFFD700);
const Color Color::Bronze(0xD99730);
const Color Color::Azure(0xF0FFFF);
const Color Color::Silver(0x808080);