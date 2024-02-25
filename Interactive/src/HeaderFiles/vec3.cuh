#ifndef VEC3_HPP
#define VEC3_HPP

#include <iostream>
#include <cmath>
#include <random>

#define FLT_MAX 3.402823466e+38F

class vec3 {
    public:
        float e[3];

        // Constructors
        __device__ __host__ vec3() : e{0, 0, 0} {}
        __device__ __host__ vec3(float e0) : e{e0, e0, e0} {}
        __device__ __host__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

        // Getters
        __device__ float x() const { return e[0]; }
        __device__ float y() const { return e[1]; }
        __device__ float z() const { return e[2]; }

        // Overloading operators
        __device__ __host__ vec3 operator -() const { return vec3(-e[0], -e[1], -e[2]); }
        __device__ __host__ float operator [](int i) const { return e[i]; }
        __device__ __host__ float& operator [](int i) { return e[i]; }
        __device__ __host__ vec3& operator +=(const vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];

            return *this;
        }
        __device__ __host__ vec3& operator *=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;

            return *this;
        }
        __device__ __host__ vec3& operator /=(const float t) {
            return *this *= 1/t;
        }

        __device__ float length() const {
            return sqrt(length_squared());
        }

        __device__ float length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        // Check if the vector is close to zero in all dimensions
        __device__ inline bool near_zero() {
            // Return true if the vector is close to zero in all dimensions
            const auto s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

};

// Type aliases for vec3
using point3 = vec3; // 3D point
using color = vec3; // RGB color

// vec3 Utility Functions
__host__ inline std::ostream& operator <<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__device__ __host__ inline vec3 operator +(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0],
                u.e[1] + v.e[1],
                u.e[2] + v.e[2]);
}

__device__ __host__ inline vec3 operator -(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0],
                u.e[1] - v.e[1],
                u.e[2] - v.e[2]);
}

__device__ __host__ inline vec3 operator *(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0],
                u.e[1] * v.e[1],
                u.e[2] * v.e[2]);
}

__device__ __host__ inline vec3 operator *(float t, const vec3& v) {
    return vec3(t * v.e[0],
                t * v.e[1],
                t * v.e[2]);
}
// This calls the above function
__device__ __host__ inline vec3 operator *(const vec3& v, float t) {
    return t * v;
}

__device__ __host__ inline vec3 operator /(const vec3& v, float t) {
    return (1/t) * v;
}

__device__ __host__ inline bool operator ==(const vec3& u, const vec3& v) {
    return (u.e[0] == v.e[0]) && (u.e[1] == v.e[1]) && (u.e[2] == v.e[2]);
}

__device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0] +
           u.e[1] * v.e[1] +
           u.e[2] * v.e[2];
}

__device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

// Reflecting a vector
__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

// Refracting a vector
// This is modeled by Snell's law
__device__ inline vec3 refract(const vec3& uv, const vec3 &n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif