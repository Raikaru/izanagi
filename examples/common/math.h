#pragma once
// Minimal math: float4x4, perspective, lookAt. Column-major, right-handed, depth 0..1.

#include <cmath>
#include <cstdint>

struct float4 {
    float x, y, z, w;
};

struct float4x4 {
    float m[16]; // column-major: m[col*4 + row]

    static float4x4 identity() {
        float4x4 r = {};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static float4x4 perspective(float fov_y, float aspect, float z_near, float z_far) {
        float4x4 r = {};
        float t = 1.0f / tanf(fov_y * 0.5f);
        r.m[0]  = t / aspect;
        r.m[5]  = t;
        r.m[10] = z_far / (z_near - z_far);   // depth 0..1
        r.m[11] = -1.0f;
        r.m[14] = z_near * z_far / (z_near - z_far);
        return r;
    }

    static float4x4 lookAt(float4 eye, float4 center, float4 up) {
        float fx = center.x - eye.x, fy = center.y - eye.y, fz = center.z - eye.z;
        float fl = sqrtf(fx * fx + fy * fy + fz * fz);
        fx /= fl; fy /= fl; fz /= fl;

        float sx = fy * up.z - fz * up.y;
        float sy = fz * up.x - fx * up.z;
        float sz = fx * up.y - fy * up.x;
        float sl = sqrtf(sx * sx + sy * sy + sz * sz);
        sx /= sl; sy /= sl; sz /= sl;

        float ux = sy * fz - sz * fy;
        float uy = sz * fx - sx * fz;
        float uz = sx * fy - sy * fx;

        float4x4 r = identity();
        r.m[0] = sx;  r.m[4] = sy;  r.m[8]  = sz;
        r.m[1] = ux;  r.m[5] = uy;  r.m[9]  = uz;
        r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz;
        r.m[12] = -(sx * eye.x + sy * eye.y + sz * eye.z);
        r.m[13] = -(ux * eye.x + uy * eye.y + uz * eye.z);
        r.m[14] = (fx * eye.x + fy * eye.y + fz * eye.z);
        return r;
    }

    static float4x4 rotate_y(float angle) {
        float4x4 r = identity();
        float c = cosf(angle), s = sinf(angle);
        r.m[0] = c;  r.m[8]  = s;
        r.m[2] = -s; r.m[10] = c;
        return r;
    }

    float4x4 operator*(const float4x4& o) const {
        float4x4 r = {};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0;
                for (int k = 0; k < 4; ++k) {
                    sum += m[k * 4 + row] * o.m[col * 4 + k];
                }
                r.m[col * 4 + row] = sum;
            }
        }
        return r;
    }

    // Slang shaders are compiled with -matrix-layout-row-major, so matrices
    // read from GPU memory are row-major. math.h is column-major internally;
    // transpose at the upload boundary before gpu_args_append.
    float4x4 transposed() const {
        float4x4 r = {};
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                r.m[col * 4 + row] = m[row * 4 + col];
            }
        }
        return r;
    }
};
