#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"
#include "random.cuh"
#define PI 3.141592653589793f

// The camera will follow a circular path around the scene. It will keep a fixed height
// and look at the center of the scene.
// Camera depedenent parameters are computed based on the elapsed time
class camera {
    public:
        __device__ camera(point3 lookfrom, point3 lookat, vec3 vup, float vfov,
                        float aspect_ratio, float aperture, float focus_dist) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta / 2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;


            // Camera orthogonal basis
            _w = unit_vector(lookfrom - lookat);
            _u = unit_vector(cross(vup, _w));
            _v = cross(_w, _u);

            _lookfrom = lookfrom;
            _lookat = lookat;
            _focus_dist = focus_dist;
            _horizontal = focus_dist * viewport_width * _u;
            _vertical = focus_dist * viewport_height * _v;
            _lower_left_corner = _lookfrom - _horizontal / 2.0f - _vertical / 2.0f - focus_dist * _w;

            _lens_radius = aperture / 2.0f;

            _movement_radius = sqrtf(powf(lookfrom.x() - lookat.x(), 2) + powf(lookfrom.z() - lookat.z(), 2));
            // Get coresponding start angle for lookfrom
            _angle = atan2f(lookfrom.z() - lookat.z(), lookfrom.x() - lookat.x());
        }
        #ifdef __CUDACC__
        __device__ ray get_ray(float s, float t, curandState *state) const {

            vec3 rd = _lens_radius * randomInUnitDisk(state);
            vec3 offset = _u * rd.x() + _v * rd.y();

            return ray(
                _lookfrom + offset,
                _lower_left_corner + s * _horizontal + t * _vertical - _lookfrom - offset
            );
        }

        __device__ void adjust_parameters(float deltaTime) {
           if (_angle >= (-PI / 2 + PI / 5)) {
				direction = false;
			} else if (_angle <= (-PI / 2 - PI / 5)) {
				direction = true;
			}

            _angle += direction ? SPEED * deltaTime : -SPEED * deltaTime;
			_lookfrom = point3(_lookat.x() + _movement_radius * cosf(_angle), _lookfrom.y(), _lookat.z() + _movement_radius * sinf(_angle));
            _w = unit_vector(_lookfrom - _lookat);

			_lower_left_corner = _lookfrom - _horizontal / 2.0f - _vertical / 2.0f - _w * _focus_dist;
        }
        #endif

    private:
        point3 _lookfrom;
        point3 _lookat;
        point3 _lower_left_corner;
        vec3 _horizontal;
        vec3 _vertical;
        vec3 _u, _v, _w;
        float _lens_radius;
        float _focus_dist;

        // Movement
        float _movement_radius;
        float _angle;
        const float SPEED = 0.05f;
        bool direction = true;


    __device__ static float degrees_to_radians(float degrees) {
        return degrees * PI / 180.0f;
    }

};

#endif

