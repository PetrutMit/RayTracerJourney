#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <glad/glad.h>
#include <GLFW/glfw3.h>

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
    }

    ~Window() {
        glfwTerminate();
    }

    void update() const {
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
};

#endif