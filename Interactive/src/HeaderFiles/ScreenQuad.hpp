#ifndef SCREEN_QUAD_HPP
#define SCREEN_QUAD_HPP

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "Render.cuh"

class ScreenQuad {

    public:
        ScreenQuad(int width, int height) : _width(width), _height(height) {
            // Geometry initialization
            GLfloat vertices[] = {
                // positions          // texture coords
                1.0f,  1.0f, 0.0f,   1.0f, 1.0f,
                1.0f, -1.0f, 0.0f,   1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
                -1.0f,  1.0f, 0.0f,  0.0f, 1.0f
            };

            GLuint indices[] = {
                0, 1, 3,
                1, 2, 3
            };

            glGenVertexArrays(1, &_vao);
            glGenBuffers(1, &_vbo);

            glBindVertexArray(_vao);

            glBindBuffer(GL_ARRAY_BUFFER, _vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            // position attribute
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);

            // texture coord attribute
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
            
            glGenBuffers(1, &_ibo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

            // Unbind
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);

            // Texture initialization
            glGenTextures(1, &_texture);
            glBindTexture(GL_TEXTURE_2D, _texture);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, _width, _height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
            glBindTexture(GL_TEXTURE_2D, 0);

            // PBO initialization
            glGenBuffers(1, &_PBO);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _PBO);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, _width * _height * 4, NULL, GL_DYNAMIC_COPY);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // CUDA-OpenGL interop 
            cudaError_t err = cudaGraphicsGLRegisterBuffer(&_cuda_resource, _PBO, cudaGraphicsRegisterFlagsNone);
            if (err != cudaSuccess) {
				std::cerr << "Error registering PBO with CUDA: " << cudaGetErrorString(err) << std::endl;
			}

            // Render initialization
            _render = new Render(_width, _height, _cuda_resource);
        }

        ~ScreenQuad() {
            delete(_render);
            glDeleteVertexArrays(1, &_vao);
            glDeleteBuffers(1, &_vbo);
            glDeleteBuffers(1, &_ibo);
            glDeleteTextures(1, &_texture);
            glDeleteBuffers(1, &_PBO);
        }

        void render_cuda_texture(GLint frameCount, GLfloat deltaTime, GLfloat camera_X, GLfloat camera_Y, GLfloat camera_Z, GLboolean denoise) {
            glBindTexture(GL_TEXTURE_2D, 0);

            _render->render(frameCount, deltaTime, camera_X, camera_Y, camera_Z, denoise);
            if (denoise)
                _render->denoise();
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _PBO);
            glBindTexture(GL_TEXTURE_2D, _texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
        }

        void render_to_screen() {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDisable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT);

			glBindVertexArray(_vao);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); // maybe go with drawArrays
		}

		GLuint get_texture() {
			return _texture;
		}

    private:
        GLuint _vao, _vbo, _ibo;
        GLuint _texture;
        GLuint _PBO;
        cudaGraphicsResource_t _cuda_resource;
        int _width, _height;
        Render *_render;
};

#endif
