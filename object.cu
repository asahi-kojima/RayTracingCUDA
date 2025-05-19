#include "object.h"

bool AABB::hit(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{
	f32 t_min_tmp = t_min;
	f32 t_max_tmp = t_max;

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

		t_min_tmp = (t0 > t_min_tmp ? t0 : t_min_tmp);
		t_max_tmp = (t1 < t_max_tmp ? t1 : t_max_tmp);
		if (t_max_tmp <= t_min_tmp)
		{
			return false;
		}
	}
	record.t = t_min_tmp;
	record.pos = ray.pointAt(record.t);
	
	vec3 center = (minPos + maxPos) / 2;
	vec3 normal_tmp = record.pos - center;
	vec3 normal = vec3(0,0,0);
	{
		vec3 pos = record.pos;
		f32 ep = 0.0001;
		if (abs(pos.getX() - maxPos.getX()) < ep)
		{
			normal.setX(1.0f);
		}
		else if (abs(pos.getY() - maxPos.getY()) < ep)
		{
			normal.setY(1.0f);
		}
		else if (abs(pos.getZ() - maxPos.getZ()) < ep)
		{
			normal.setZ(1.0f);
		}
		else if (abs(pos.getX() - minPos.getX()) < ep)
		{
			normal.setX(-1.0f);
		}
		else if (abs(pos.getY() - minPos.getY()) < ep)
		{
			normal.setY(-1.0f);
		}
		else
		{
			normal.setZ(-1.0f);
		}
	}
	//normal[abs(normal_tmp.maxElementIndex()) - 1] = (normal_tmp.maxElementIndex() > 0 ? 1 : -1);
	record.normal = normal;
	record.material = this->material;
	return true;
}


bool Triangle::hit(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{


	const vec3 p1 = mVertices[1] - mVertices[0];
	const vec3 p2 = mVertices[2] - mVertices[0];
	const vec3 v0ToO = ray.origin() - mVertices[0];

	const vec3 a0 = -ray.direction();
	const vec3 a1 = p1;
	const vec3 a2 = p2;

	const vec3 cross1x2 = vec3::cross(a1, a2);
	const vec3 cross2x0 = vec3::cross(a2, a0);
	const vec3 cross0x1 = vec3::cross(a0, a1);

	const f32 det = dot(cross1x2, a0);
	if (det == 0.0)
	{
		return false;
	}

	const f32 t = dot(cross1x2, v0ToO) / det;
	const f32 alpha = dot(cross2x0, v0ToO) / det;
	const f32 beta = dot(cross0x1, v0ToO) / det;

	if (!(t > t_min && t < t_max && alpha + beta < 1 && alpha > 0 && beta > 0))
	{
		return false;
	}


	record.t = t;
	record.pos = ray.pointAt(t);
	record.normal = mNormal * (mIsCulling ? 1 : (dot(ray.direction(), mNormal) < 0) ? 1 : -1);
	record.material = mMaterial;
	return true;
}

AABB Triangle::calcAABB()
{
	return mAABB;
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
