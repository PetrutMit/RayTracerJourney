#ifndef RTW_STB_IMAGE_HPP
#define RTW_STB_IMAGE_HPP

// stb_image is no longer present in this implement
// because it affects the percentage of the CUDA code presented by GIT

// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
    // Microsoft Visual C++ Compiler
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Restore warning levels.
#ifdef _MSC_VER
    // Microsoft Visual C++ Compiler
    #pragma warning (pop)
#endif

#endif

