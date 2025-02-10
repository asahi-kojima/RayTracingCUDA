#include "object.h"

bool AABB::hit(const Ray &r, const f32 t_min, const f32 t_max, HitRecord &record)
{
	assert(0);
	return false;
}

bool Sphere::hit(const Ray &r, const f32 t_min, const f32 t_max, HitRecord &record)
{
	const vec3 &direction = r.direction();
	vec3 oc = r.origin() - center;
	f32 a = dot(direction, direction);
	f32 b = 2 * dot(direction, oc);
	f32 c = dot(oc, oc) - radius * radius;
	f32 D = b * b - 4 * a * c;

	bool isHit = false;
	f32 tmp = 0.0f;

	if (D > 0)
	{
		const f32 root_of_D = sqrtf(D);
		const f32 inv_2a = 1 / (2.0f * a);

		tmp = (-b - root_of_D) * inv_2a;
		if (tmp < t_max && tmp > t_min)
		{
			isHit = true;
		}

		if (!isHit)
		{
			tmp = (-b + root_of_D) * inv_2a;
			if (tmp < t_max && tmp > t_min)
			{
				isHit = true;
			}
		}
	}

	record.t = tmp;
	record.pos = r.pointAt(tmp);
	record.normal = (record.pos - center) *(1.0f / radius);
	record.material = this->material;

	return isHit;
}

AABB Sphere::calcAABB()
{
	const vec3 v_min = center - fabsf(radius);
	const vec3 v_max = center + fabsf(radius);
	return AABB(v_min, v_max);
}

bool AABB::isIntersecting(const Ray &ray,  f32 t_min,  f32 t_max) const
{
	const vec3 &origin = ray.origin();
	const vec3 &direction = ray.direction();
	for (u32 i = 0; i < 3; i++)
	{
		const f32 inv = 1.0f / direction[i];
		const f32 ith_origin = origin[i];
		f32 t0 = (minPos[i] - ith_origin) * inv;
		f32 t1 = (maxPos[i] - ith_origin) * inv;

		if (inv < 0.0f)
		{
			aoba::swap(t0, t1);
		}

		t_min = (t0 > t_min ? t0 : t_min);
		t_max = (t1 < t_max ? t1 : t_max);
		if (t_max <= t_min)
		{
			return false;
		}
	}

	return true;
}
