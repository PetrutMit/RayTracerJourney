#include "HeaderFiles/Window.hpp"

int main() {
    Window *window = new Window(800, 600);

    window->loop();

    return 0;
}