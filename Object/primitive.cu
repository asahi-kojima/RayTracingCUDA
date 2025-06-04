#include "object.h"

bool AABB::isHitInLocalSpace(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{
#if 1
	const Vec3 center = (maxPos + minPos) * 0.5f;
	const Vec3 extention = (maxPos - minPos) * 0.5f;
	const f32 extention_x = extention[0];
	const f32 extention_y = extention[1];
	const f32 extention_z = extention[2];
	const Vec3 vertex_list[4] = {
		center + Vec3(+extention_x, +extention_y, +extention_z),
		center + Vec3(-extention_x, +extention_y, +extention_z),
		center + Vec3(+extention_x, -extention_y, +extention_z),
		center + Vec3(+extention_x, +extention_y, -extention_z)};

	const size_t index_list[4 * 3] = {0,1,2,3,   0,2,3,1,   0,3,1,2};

	f32 current_min_t = MAXFLOAT;
	Vec3 normal;
	u32 counter = 0;
	bool isAnyHit = false;
	for (u32 i = 0; i < 3; i++)
	{
		const u32 offset = 4 * i;
		const Vec3 positive_origin = vertex_list[index_list[offset + 0]];
		const Vec3 negative_origin = vertex_list[index_list[offset + 1]];
		const Vec3 other_vertex_for_edge1 = vertex_list[index_list[offset + 2]];
		const Vec3 other_vertex_for_edge2 = vertex_list[index_list[offset + 3]];

		const Vec3 p1 = other_vertex_for_edge1 - positive_origin;
		const Vec3 p2 = other_vertex_for_edge2 - positive_origin;
		const Vec3 v0ToO_list[2] = {ray.origin() - positive_origin, ray.origin() - negative_origin};

		const Vec3 a0 = -ray.direction();
		const Vec3 a1 = p1;
		const Vec3 a2 = p2;

		const Vec3 cross1x2 = Vec3::cross(a1, a2);
		const Vec3 cross2x0 = Vec3::cross(a2, a0);
		const Vec3 cross0x1 = Vec3::cross(a0, a1);

		const f32 det = dot(cross1x2, a0);
		if (det == 0.0)
		{
			continue;
		}

		for (s32 j = 0; j < 2; j++)
		{
			const Vec3& v0ToO = v0ToO_list[j]; 
			const f32 t = dot(cross1x2, v0ToO) / det;
			const f32 alpha = dot(cross2x0, v0ToO) / det;
			const f32 beta = dot(cross0x1, v0ToO) / det;

			if (!(t > t_min && t < t_max && alpha > 0 && beta > 0 && alpha < 1 && beta < 1))
			{
				continue;
			}

			isAnyHit = true;
			counter++;
			if (t < current_min_t)
			{
				current_min_t = t;
				normal = Vec3(0, 0, 0);
				normal[i] = 1 - 2 * j;
			}

		}
	}
	if (!isAnyHit)
	{
		return false;
	}
	
	record.t = current_min_t;	
	record.pos = ray.pointAt(current_min_t);
	record.normal = normal;
	record.material = this->material;
	return true;


#else


	//========================================================
	f32 t_min_tmp = t_min;
	f32 t_max_tmp = t_max;

	const Vec3 &origin = ray.origin();
	const Vec3 &direction = ray.direction();

	bool isRayOriginInThis = true;
	u32 max_t0_index = 0;
	u32 min_t1_index = 0;
	f32 max_t0_value = -MAXFLOAT;
	f32 min_t1_value = MAXFLOAT;

	for (u32 i = 0; i < 3; i++)
	{
		const f32 inv_direction = 1.0f / direction[i];
		const f32 ith_origin = origin[i];
		f32 t0 = (minPos[i] - ith_origin) * inv_direction;
		f32 t1 = (maxPos[i] - ith_origin) * inv_direction;
		
		if (inv_direction < 0.0f)
		{
			aoba::swap(t0, t1);
		}


		if (t0 > max_t0_value)
		{
			max_t0_value = t0;
			max_t0_index = i;
		}
		if (t1 < min_t1_value)
		{
			min_t1_value = t1;
			min_t1_index = i;
		}


		if (t0 > 0)
		{
			isRayOriginInThis = false;
		}

		t_min_tmp = (t0 > t_min_tmp ? t0 : t_min_tmp);
		t_max_tmp = (t1 < t_max_tmp ? t1 : t_max_tmp);
		if (t_max_tmp <= t_min_tmp)
		{
			return false;
		}
	}



	const f32 t = (isRayOriginInThis ? min_t1_value : max_t0_value);
	const Vec3 pos = ray.pointAt(t);
	
	
	
	
	Vec3 normal = Vec3(0,0,0);
	{
		if (isRayOriginInThis)
		{
			const f32 diff = pos[min_t1_index] - mCenter[min_t1_index];
			normal[min_t1_index] = (diff > 0 ? 1 : -1);
		}
		else
		{
			const f32 diff = pos[max_t0_index] - mCenter[max_t0_index];
			normal[max_t0_index] = (diff > 0 ? 1 : -1);
		}
	}
	
		
	record.t = t;	
	record.pos = pos;
	record.normal = normal;
	record.material = this->material;
	return true;
#endif
}


bool Triangle::isHitInLocalSpace(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{
	const Vec3 p1 = mVertices[1] - mVertices[0];
	const Vec3 p2 = mVertices[2] - mVertices[0];
	const Vec3 v0ToO = ray.origin() - mVertices[0];

	const Vec3 a0 = -ray.direction();
	const Vec3 a1 = p1;
	const Vec3 a2 = p2;

	const Vec3 cross1x2 = Vec3::cross(a1, a2);
	const Vec3 cross2x0 = Vec3::cross(a2, a0);
	const Vec3 cross0x1 = Vec3::cross(a0, a1);

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

AABB Triangle::getAABB()
{
	return mAABB;
}

bool Sphere::isHitInLocalSpace(const Ray &r, const f32 t_min, const f32 t_max, HitRecord &record)
{
	const Vec3 &direction = r.direction();
	Vec3 oc = r.origin();
	f32 a = dot(direction, direction);
	f32 b = 2 * dot(direction, oc);
	f32 c = dot(oc, oc) - 1.0f;
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
	record.normal = (record.pos - center);
	record.material = this->material;

	return isHit;
}

AABB Sphere::getAABB()
{
	const Vec3 v_min = center - fabsf(radius);
	const Vec3 v_max = center + fabsf(radius);
	return AABB(v_min, v_max);
}

bool AABB::isIntersecting(const Ray &ray,  f32 t_min,  f32 t_max) const
{
	const Vec3 &origin = ray.origin();
	const Vec3 &direction = ray.direction();
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


AABB AABB::tranformWith(const Transform& transform) const
{
	AABB aabb;


	return aabb;
}
