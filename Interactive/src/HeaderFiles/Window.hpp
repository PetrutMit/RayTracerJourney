#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "ScreenQuad.hpp"
#include "Shader.hpp"

class Window {

    public:
        Window(int width, int height) : _width(width), _height(height) {
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

            _quad = new ScreenQuad(_width, _height);

            _shader = new Shader("./Shaders/vertex.glsl", "./Shaders/fragment.glsl");
        }

        ~Window() {
            glfwTerminate();
        }

        void update() const {
            // Render the textured quad
            _shader->use();
            // Activate the texture
            GLuint texture = _quad->get_texture();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glUniform1i(glGetUniformLocation(_shader->ID, "screenTexture"), 0);

            _quad->render_to_screen();

            // Draw Call
            glfwSwapBuffers(_window);
            glfwPollEvents();
        }

        void process_input() const {
            if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(_window, true);
            }
        }

        void loop() const {
            while (!glfwWindowShouldClose(_window)) {
                process_input();
                update();
            }
        }



    private:
        GLFWwindow* _window;
        int _width;
        int _height;

        ScreenQuad* _quad;
        Shader* _shader;
};

#endif