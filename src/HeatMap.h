// HeatMap.h
// Dan Wolf

#ifndef HOTHOTHOT_HEATMAP_H
#define HOTHOTHOT_HEATMAP_H

#include "ObjectSnapshot.h"

void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height);

struct Pixel {
    unsigned char R;
    unsigned char G;
    unsigned char B;
    unsigned char A;
};

class HeatMap {
public:

    explicit HeatMap(const ObjectSnapshot& object, bool normalize);

    ~HeatMap();

    void generate() const;
    Pixel colorFor(float value) const;
private:
    ObjectSnapshot object;
    bool normalize;
};


#endif //HOTHOTHOT_HEATMAP_H
