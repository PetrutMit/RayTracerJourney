#include "HeaderFiles/Window.hpp"

int main() {
    // Create a window
    Window *window = new Window(800, 600);

    window->loop();

    return 0;
}