// HeatMap.cpp
// Dan Wolf

#include <iostream>
#include "HeatMap.h"
#include "lodepng.h"

HeatMap::~HeatMap() = default;

HeatMap::HeatMap(const ObjectSnapshot &object, bool normalize) {
    this->object = object;
    this->normalize = normalize;
}

Pixel HeatMap::colorFor(float value) const {
    float v = this->normalize ? (value - this->object.min) / this->object.spread : value;
    // using color scheme from https://www.schemecolor.com/rgb-blue-to-red-gradient.php
    if (v < (normalize ? .167 : 16.7)) return Pixel{.R = 0x03, .G=0x02, .B=0xFC, .A=0xFF};
    else if (v < (normalize ? .333 : 33.3)) return Pixel{.R = 0x2A, .G=0x00, .B=0xD5, .A=0xFF};
    else if (v < (normalize ? .5 : 50)) return Pixel{.R = 0x63, .G=0x00, .B=0x9E, .A=0xFF};
    else if (v < (normalize ? .667 : 66.7)) return Pixel{.R = 0xA1, .G=0x01, .B=0x5D, .A=0xFF};
    else if (v < (normalize ? .833 : 83.3)) return Pixel{.R = 0xD8, .G=0x00, .B=0x27, .A=0xFF};
    else return Pixel{.R = 0xFE, .G=0x00, .B=0x02, .A=0xFF};
}

void HeatMap::generate() const {
    const char *filename = "heat.png";
    unsigned width = this->object.array.size(), height = 200;
    std::vector<unsigned char> image;
    image.resize(width * height * 4);
    for (unsigned x = 0; x < width; x++)
        for (unsigned y = 0; y < height; y++) {
            auto value = this->object.array[x];
            auto pixel = this->colorFor(value);
            image[4 * width * y + 4 * x + 0] = pixel.R;
            image[4 * width * y + 4 * x + 1] = pixel.G;
            image[4 * width * y + 4 * x + 2] = pixel.B;
            image[4 * width * y + 4 * x + 3] = pixel.A;
        }
    encodeOneStep(filename, image, width, height);
}

// this function is from a third party example from lodepng, see 3rd party sources
// https://raw.githubusercontent.com/lvandeve/lodepng/master/examples/example_encode.cpp
// Encode from raw pixels to disk with a single function call
// The image argument has width * height RGBA pixels or width * height * 4 bytes
void encodeOneStep(const char *filename, std::vector<unsigned char> &image, unsigned width, unsigned height) {
    //Encode the image
    unsigned error = lodepng::encode(filename, image, width, height);

    //if there's an error, display it
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}