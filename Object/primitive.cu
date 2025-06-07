#include "primitive.h"

bool AABB::isHit(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{
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

		const f32 det = Vec3::dot(cross1x2, a0);
		if (det == 0.0)
		{
			continue;
		}

		for (s32 j = 0; j < 2; j++)
		{
			const Vec3& v0ToO = v0ToO_list[j]; 
			const f32 t = Vec3::dot(cross1x2, v0ToO) / det;
			const f32 alpha = Vec3::dot(cross2x0, v0ToO) / det;
			const f32 beta = Vec3::dot(cross0x1, v0ToO) / det;

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
	record.position = ray.pointAt(current_min_t);
	record.normal = normal;
	record.material = this->material;
	return true;
}

AABB AABB::tranformWith(const Mat4& transformMat) const
{
	const f32 x_min = minPos[0];
	const f32 y_min = minPos[1];
	const f32 z_min = minPos[2];
	const f32 x_max = maxPos[0];
	const f32 y_max = maxPos[1];
	const f32 z_max = maxPos[2];
	Vec4 vertex[8];
	Vec3 transformed_vertex[8];
	for (u32 i = 0; i < 8; i++)
	{
		//値をセット
		vertex[i][0] = (((i >> 0) & 0x1) == 1 ? x_min : x_max);
		vertex[i][1] = (((i >> 1) & 0x1) == 1 ? y_min : y_max);
		vertex[i][2] = (((i >> 2) & 0x1) == 1 ? z_min : z_max);
		vertex[i][3] = 1.0f;

		//トランスフォームを行う
		transformed_vertex[i] = (transformMat * vertex[i]).extractXYZ();
	}

	f32 new_x_min = MAXFLOAT;
	f32 new_y_min = MAXFLOAT;
	f32 new_z_min = MAXFLOAT;
	f32 new_x_max = -MAXFLOAT;
	f32 new_y_max = -MAXFLOAT;
	f32 new_z_max = -MAXFLOAT;
	for (u32 i = 0; i < 8; i++)
	{
		new_x_min = min(new_x_min, transformed_vertex[i][0]);
		new_y_min = min(new_y_min, transformed_vertex[i][1]);
		new_z_min = min(new_z_min, transformed_vertex[i][2]);
		new_x_max = max(new_x_max, transformed_vertex[i][0]);
		new_y_max = max(new_y_max, transformed_vertex[i][1]);
		new_z_max = max(new_z_max, transformed_vertex[i][2]);
	}

	return AABB(Vec3(new_x_min, new_y_min, new_z_min), Vec3(new_x_max, new_y_max, new_z_max));
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



bool Sphere::isHit(const Ray &r, const f32 t_min, const f32 t_max, HitRecord &record)
{
	const Vec3 &direction = r.direction();
	Vec3 oc = r.origin();
	f32 a = Vec3::dot(direction, direction);
	f32 b = 2 * Vec3::dot(direction, oc);
	f32 c = Vec3::dot(oc, oc) - 1.0f;
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
	record.position = r.pointAt(tmp);
	record.normal = record.position;

	return isHit;
}

AABB Sphere::getAABB()
{
	return mAABB;
}




