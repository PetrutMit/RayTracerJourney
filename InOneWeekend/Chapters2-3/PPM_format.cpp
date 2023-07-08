/*
* Testing the PPM format
* Printed the output in a file called PPM_format.ppm to get the actual image
* This image can be viewed in any image viewer
* Also add some sort of progress indicator which is in our case a simple
* std::err print statement
*
* Pixels are printed from left to right, from top to bottom
* RGB format which is at first normalized to [0, 1]
*/
#include <iostream>
#include <fstream>

int main() {

    // The output of our program is a PPM image
    std::ofstream ppm_file("PPM_format.ppm");

    // Seting our resolution -> 256x256
    const int image_width = 256;
    const int image_height = 256;

    // Printing the PPM header
    ppm_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // Printing the pixels
    for (int j = image_height - 1; j >= 0; j --) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i ++) {
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.25;

            // Converting the color to RGB
            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            // Printing the pixels
            ppm_file << ir << '-' << ig << '-' << ib << ' ';
        }
        ppm_file << '\n';
    }

    std::cerr << "\nDone.\n";
}