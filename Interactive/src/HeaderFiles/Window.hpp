#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "ScreenQuad.hpp"
#include "Shader.hpp"

class Window {

public:
    Window(int width, int height) : _width(width), _height(height), _last_frame(glfwGetTime()), _enable_denoising(true) {
            if (!glfwInit()) {
                throw "Failed to initialize GLFW";
            }

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

            _window = glfwCreateWindow(_width, _height, "Interactive", NULL, NULL);
            if (!_window) {
                glfwTerminate();
                throw "Failed to create window";
            }

            glfwMakeContextCurrent(_window);

            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                glfwTerminate();
                throw "Failed to initialize GLAD";
            }

            glViewport(0, 0, _width, _height);
            GLFWframebuffersizefun framebuffer_size_callback = [](GLFWwindow* window, int width, int height) {
				static_cast<Window*>(glfwGetWindowUserPointer(window))->framebuffer_size_callback(window, width, height);
			};
            glfwSetWindowUserPointer(_window, this);
            glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);

            _quad = new ScreenQuad(_width, _height);

            _shader_render = new Shader("./Shaders/render_pass_vertex.glsl",
                    					"./Shaders/render_pass_frag.glsl");

            _title = new char[100];
        }

        ~Window() {
            glfwTerminate();
            delete(_quad);
            delete(_shader_render);
        }

        void update() {
            // Compute Delta Time
            GLfloat current_frame = glfwGetTime();
            GLfloat delta_time = current_frame - _last_frame;
            _last_frame = current_frame;
            
            // Set the window title
            sprintf(_title, "Path Tracer - %.2f FPS", 1.0f / delta_time);
            glfwSetWindowTitle(_window, _title);

            // Render the CUDA texture
            _quad->render_cuda_texture(delta_time, _enable_denoising);
            
            _shader_render->use();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, _quad->get_texture());
            glUniform1i(glGetUniformLocation(_shader_render->ID, "accumulatedTexture"), 0);
            _quad->render_to_screen();

            // Draw Call
            glfwSwapBuffers(_window);
            glfwPollEvents();
        }

        void process_input() {
            if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(_window, true);
            }

            if (glfwGetKey(_window, GLFW_KEY_P) == GLFW_PRESS) {
                _enable_denoising = _enable_denoising ? false : true;
            }
        }

        void loop() {
           while (!glfwWindowShouldClose(_window)) {
                process_input();
                update();
            }
        }

        void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
            _width = width;
            _height = height;
			glViewport(0, 0, width, height);
		}
    
    private:
        // Window
        GLFWwindow* _window;
        int _width;
        int _height;
        GLfloat _last_frame;
        ScreenQuad* _quad;
        char* _title;

        Shader* _shader_render;
        GLboolean _enable_denoising;
};

#endif
