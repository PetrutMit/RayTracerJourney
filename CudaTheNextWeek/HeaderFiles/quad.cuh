#ifndef QUAD_CUH
#define QUAD_CUH

#include "hittable.cuh"
#include "vec3.cuh"

class quad : public hittable {
    public:
        __device__ quad() {}

        __device__ quad(const vec3& Q, const vec3& u, const vec3& v, material *mat_ptr) : _Q(Q), _u(u), _v(v), _mat_ptr(mat_ptr) {
            vec3 n = cross(u, v);
            _normal = unit_vector(n);
            _D = dot(_normal, Q);
            _w = n / dot(n, n);

            set_bounding_box();
        }

        __device__ virtual void set_bounding_box() {
            _bbox = aabb(_Q, _Q + _u + _v).pad();
        }

        __device__ virtual bool is_interior(float a, float b, hit_record &rec) const {
            if ((a < 0) || (1 < a) || (b < 0) || (1 < b)) {
                return false;
            }

            rec.u = a;
            rec.v = b;
            return true;
        }

        __device__ aabb bounding_box() const override {
            return _bbox;
        }

        __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
            float denom = dot(_normal, r.direction());

            // No hit if the ray is parallel to the plane
            if (fabsf(denom) < 1e-6) {
                return false;
            }

            // Return false if the hit point parameter t is outside the interval
            float t = (_D - dot(_normal, r.origin())) / denom;
            if (!ray_t.contains(t)) {
                return false;
            }

            // Determine if the point lies within the quad using planar coordinates
            vec3 P = r.at(t);
            vec3 planar_hitpt_vector = P - _Q;
            float alpha = dot(_w, cross(planar_hitpt_vector, _v));
            float beta = dot(_w, cross(_u, planar_hitpt_vector));

            if (!is_interior(alpha, beta, rec)) {
                return false;
            }

            rec.t = t;
            rec.p = P;
            rec.mat_ptr = _mat_ptr;
            rec.set_face_normal(r, _normal);

            return true;
        }

    private:
        vec3 _Q;
        vec3 _u, _v;
        material *_mat_ptr;
        aabb _bbox;
        vec3 _normal;
        float _D;
        vec3 _w;
};

inline __device__ hittable_list *box(const vec3 &a, const vec3 &b, material *mat_ptr) {
    // Computes the 3D box (six sides) that contains the 2 opposite points a and b
    hittable **list = new hittable*[6];

    vec3 min = vec3(fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()));
    vec3 max = vec3(fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()));

    vec3 dx = vec3(max.x() - min.x(), 0, 0);
    vec3 dy = vec3(0, max.y() - min.y(), 0);
    vec3 dz = vec3(0, 0, max.z() - min.z());

    list[0] = new quad(vec3(min.x(), min.y(), max.z()), dx, dy, mat_ptr); //front
    list[1] = new quad(vec3(max.x(), min.y(), max.z()), -dz, dy, mat_ptr); //right
    list[2] = new quad(vec3(max.x(), min.y(), min.z()), -dx, dy, mat_ptr); //back
    list[3] = new quad(vec3(min.x(), min.y(), min.z()), dz, dy, mat_ptr); //left
    list[4] = new quad(vec3(min.x(), max.y(), max.z()), dx, -dz, mat_ptr); //top
    list[5] = new quad(vec3(min.x(), min.y(), max.z()), dx, dz, mat_ptr); //bottom

    return new hittable_list(list, 6);
}

#endif