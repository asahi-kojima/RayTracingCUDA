#include <stdio.h>
#include "color.h"
#include "util.h"


Color::Color(const f32 r, const f32 g, const f32 b, const f32 alpha)
	: mRGBA{ r, g, b,alpha }
{
	Color::clamp();
}

Color::Color(const Vec3& rgb, const f32 alpha)
	: Color(rgb[0], rgb[1], rgb[1], alpha)
{
}



Color::Color(const u32 hexadecimal)
	: mRGBA{ static_cast<f32>((hexadecimal & 0xFF0000) >> 16) / 255.0f,  static_cast<f32>((hexadecimal & 0x00FF00) >> 8) / 255.0f, static_cast<f32>((hexadecimal & 0x0000FF) >> 0) / 255.0f, 1.0f }
{
}

f32 Color::operator[](size_t i) const
{
#ifdef _DEBUG
	if (!(i >= 0 && i < 4))
	{
		assert(0);
	}
#endif

	return mRGBA[i];
}

f32& Color::operator[](size_t i)
{
#ifdef _DEBUG
	if (!(i >= 0 && i < 4))
	{
		assert(0);
	}
#endif

	return mRGBA[i];
}

__device__ __host__ f32& Color::r()
{
	return mRGBA[0];
}

__device__ __host__ f32& Color::g()
{
	return mRGBA[1];
}

__device__ __host__ f32& Color::b()
{
	return mRGBA[2];
}

__device__ __host__ f32& Color::a()
{
	return mRGBA[3];
}

__device__ __host__ f32 Color::r() const
{
	return mRGBA[0];
}

__device__ __host__ f32 Color::g() const
{
	return mRGBA[1];
}

__device__ __host__ f32 Color::b() const
{
	return mRGBA[2];
}

__device__ __host__ f32 Color::a() const
{
	return mRGBA[3];
}


Color Color::operator*(const Color& rhs) const
{
	const Color& lhs = *this;
	
	f32 color[4];
	for (u32 i = 0; i < 4; i++)
	{
		color[i] = lhs.mRGBA[i] * rhs.mRGBA[i];
	}

	return Color(color[0], color[1], color[2], color[3]);
}

Color Color::operator*(const f32 value) const
{
	return Color(mRGBA[0] * value, mRGBA[1] * value, mRGBA[2] * value, mRGBA[3] * value);
}


Color& Color::operator*=(const Color& rhs)
{
	for (u32 i = 0; i < 4; i++)
	{
		mRGBA[i] *= rhs.mRGBA[i];
	}

	return *this;
}

Color& Color::operator*=(const f32 value)
{
	for (u32 i = 0; i < 3; i++)
	{
		mRGBA[i] *= mRGBA[i] * value;
	}

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

Color Color::random()
{
	return Color(RandomGenerator::uniform_int(0, 0xFFFFFF));
}


void Color::clamp()
{
	for (u32 i = 0; i < 3; i++)
	{
		f32 tmp = mRGBA[i];
		tmp = (tmp > 1.0f ? 1.0f : tmp);
		tmp = (tmp < 0.0f ? 0.0f : tmp);
		mRGBA[i] = tmp;
	}
}

bool Color::isNan()
{
	return (isnan(mRGBA[0]) || isnan(mRGBA[1]) || isnan(mRGBA[2]) || isnan(mRGBA[3]));
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