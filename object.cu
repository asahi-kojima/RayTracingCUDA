#include "object.h"

bool AABB::hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{
	return false;
}

bool Sphere::hit(const Ray& r, const f32 t_min, const f32 t_max, HitRecord& record)
{
	vec3 oc = r.origin() - center;
	f32 a = dot(r.direction(), r.direction());
	f32 b = 2 * dot(r.direction(), oc);
	f32 c = dot(oc, oc) - radius * radius;
	f32 D = b * b - 4 * a * c;

	if (D > 0)
	{
		//��������
		if (c < 0)
		{
			f32 tmp = (-b + sqrtf(D)) / (2.0f * a);
			if (tmp < t_max && tmp > t_min)
			{
				record.t = tmp;
				record.pos = r.pointAt(tmp);
				record.normal = (record.pos - center) / radius;
				record.material = this->material;
				return true;
			}
		}
		else
		{
			f32 tmp = (-b - sqrtf(D)) / (2.0f * a);
			if (tmp < t_max && tmp > t_min)
			{
				record.t = tmp;
				record.pos = r.pointAt(tmp);
				record.normal = (record.pos - center) / radius;
				record.material = this->material;
				return true;
			}

			tmp = (-b + sqrtf(D)) / (2.0f * a);
			if (tmp < t_max && tmp > t_min)
			{
				record.t = tmp;
				record.pos = r.pointAt(tmp);
				record.normal = (record.pos - center) / radius;
				record.material = this->material;
				return true;
			}
		}
	}

	return false;
}

AABB Sphere::calcAABB()
{
	const vec3 v_min = center - radius;
	const vec3 v_max = center + radius;
	return AABB(v_min, v_max);
}

AABB wraping(AABB lhs, AABB rhs)
{

	return AABB{};
}

bool isIntersecting(const AABB& aabb, const Ray& ray, f32 t_min, f32 t_max)
{
	const vec3& min_pos = aabb.getMinPos();
	const vec3& max_pos = aabb.getMaxPos();
	const vec3& origin = ray.origin();
	const vec3& direction = ray.direction();

	auto intersection_per_axis = [&t_min, &t_max](f32 min_pos, f32 max_pos, f32 origin, f32 direction, f32& t_min_x, f32& t_max_x)->bool
		{
			t_min_x = std::min((min_pos - origin) / direction, (max_pos - origin) / direction);
			t_max_x = std::max((min_pos - origin) / direction, (max_pos - origin) / direction);

			if (isnan(t_min_x) || isnan(t_max_x))
			{
				return false;
			}

			t_min = fmaxf(t_min, t_min_x);
			t_max = fminf(t_max, t_max_x);

			if (t_max <= t_min)
			{
				return false;
			}

			return true;
		};
	
	//x������
	f32 t_min_x, t_max_x;
	{
		f32 min_pos_x = min_pos.getX();
		f32 max_pos_x = max_pos.getX();
		f32 origin_x = origin.getX();
		if (!intersection_per_axis(min_pos_x, max_pos_x, origin_x, direction.getX(), t_min_x, t_max_x))
		{
			return false;
		}
	}

	f32 t_min_y, t_max_y;
	{
		f32 min_pos_y = min_pos.getY();
		f32 max_pos_y = max_pos.getY();
		f32 origin_y = origin.getY();
		if (!intersection_per_axis(min_pos_y, max_pos_y, origin_y, direction.getY(), t_min_y, t_max_y))
		{
			return false;
		}
	}

	f32 t_min_z, t_max_z;
	{
		f32 min_pos_z = min_pos.getZ();
		f32 max_pos_z = max_pos.getZ();
		f32 origin_z = origin.getZ();
		if (!intersection_per_axis(min_pos_z, max_pos_z, origin_z, direction.getZ(), t_min_z, t_max_z))
		{
			return false;
		}
	}


	return true;
}
