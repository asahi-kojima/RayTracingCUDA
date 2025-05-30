#include <cassert>
#include <math.h>
#include <random>
#include "vector.h"
#include "util.h"


Vec3 Vec4::extractXYZ() const
{
	return Vec3(mElements[0], mElements[1], mElements[2]);
}