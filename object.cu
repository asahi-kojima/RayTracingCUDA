#include "object.h"

bool AABB::hit(const Ray &ray, const f32 t_min, const f32 t_max, HitRecord &record)
{
	{
		const vec3 center = (maxPos + minPos) * 0.5f;
		const vec3 extention = (maxPos - minPos) * 0.5f;
		const f32 x = extention[0];
		const f32 y = extention[1];
		const f32 z = extention[2];
		const vec3 vertex_list[8] = {
		center + vec3(+x, +y, +z),
		center + vec3(-x, +y, +z),
		center + vec3(+x, -y, +z),
		center + vec3(-x, -y, +z),
		center + vec3(+x, +y, -z),
		center + vec3(-x, +y, -z),
		center + vec3(+x, -y, -z),
		center + vec3(-x, -y, -z)};

		const size_t index_list[18] = {3,2,1,    2, 6, 0,   7, 3, 5,   6, 7, 4,   1, 0, 5,   7, 6, 3};
		const vec3 normal_list[6] = {vec3(0, 0, 1), vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0), vec3(0, -1, 0)};

		f32 current_min_t = MAXFLOAT;
		vec3 normal;
		u32 counter = 0;
		bool isAnyHit = false;
		for (u32 i = 0; i < 6; i++)
		{
			const u32 offset = 3 * i;
			const vec3 vertex0 = vertex_list[index_list[offset + 0]];
			const vec3 vertex1 = vertex_list[index_list[offset + 1]];
			const vec3 vertex2 = vertex_list[index_list[offset + 2]];

			const vec3 p1 = vertex1 - vertex0;
			const vec3 p2 = vertex2 - vertex0;
			const vec3 v0ToO = ray.origin() - vertex0;

			const vec3 a0 = -ray.direction();
			const vec3 a1 = p1;
			const vec3 a2 = p2;

			const vec3 cross1x2 = vec3::cross(a1, a2);
			const vec3 cross2x0 = vec3::cross(a2, a0);
			const vec3 cross0x1 = vec3::cross(a0, a1);

			const f32 det = dot(cross1x2, a0);
			if (det == 0.0)
			{
				continue;
			}

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
				normal = normal_list[i];
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
	}




	//========================================================
	f32 t_min_tmp = t_min;
	f32 t_max_tmp = t_max;

	const vec3 &origin = ray.origin();
	const vec3 &direction = ray.direction();

	bool isRayOriginInThis = true;
	u32 max_t0_index = 0;
	u32 min_t1_index = 0;
	f32 max_t0_value = -MAXFLOAT;
	f32 min_t1_value = MAXFLOAT;
	// f32 t0_values_deb[3];
	// f32 t1_values_deb[3];
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


		// t0_values_deb[i] = t0;
		// t1_values_deb[i] = t1;
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
	const vec3 pos = ray.pointAt(t);
	
	
	
	
	vec3 normal = vec3(0,0,0);
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
	
	// //Debug
	// {
	// 	auto max_index = 0;
	// 	auto max_value = abs(pos[0]);
	// 	for (u32 i =1 ; i < 3;i++)
	// 	{
	// 		if (abs(pos[i]) > max_value)
	// 		{
	// 			max_value = abs(pos[i]);
	// 			max_index = i;
	// 		}
	// 	}
	// 	vec3 testNormal(0, 0, 0);
	// 	testNormal[max_index] = (pos[max_index] > 0 ? 1 : -1);
	// 	if ((normal - testNormal).lengthSquared() > 1e-3)
	// 	{
	// 		if (isRayOriginInThis)
	// 		printf("(%f, %f, %f) != (%f, %f, %f)\n(%f, %f, %f)\n%d : [%f, %f, %f]\n", 
	// 			normal[0],normal[1],normal[2],
	// 			testNormal[0],testNormal[1],testNormal[2], 
	// 			pos[0], pos[1], pos[2],
	// 			(isRayOriginInThis?min_t1_index : -max_t0_index - 1),
	// 			t1_values_deb[0], t1_values_deb[1], t1_values_deb[2]);
	// 		else
	// 		printf("(%f, %f, %f) != (%f, %f, %f)\npos(%f, %f, %f)\n%d : [%f, %f, %f]\n  : [%f, %f, %f]\nminPos{%f, %f, %f} - origin{%f, %f, %f} : {%f, %f, %f}\n", 
	// 			normal[0],normal[1],normal[2],
	// 			testNormal[0],testNormal[1],testNormal[2], 
	// 			pos[0], pos[1], pos[2],
	// 			(isRayOriginInThis?min_t1_index : -max_t0_index - 1),
	// 			t0_values_deb[0], t0_values_deb[1], t0_values_deb[2],
	// 			t1_values_deb[0], t1_values_deb[1], t1_values_deb[2],
	// 		minPos[0],minPos[1],minPos[2], 
	// 		origin[0], origin[1], origin[2],
	// 		1.0f / direction[0], 1.0f / direction[1], 1.0f / direction[2]);
	// 	}	
	// }
					
	record.t = t;	
	record.pos = pos;
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
