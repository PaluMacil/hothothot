// ObjectSnapshot.cpp
// Dan Wolf

#include "ObjectSnapshot.h"


ObjectSnapshot::ObjectSnapshot() = default;

ObjectSnapshot::ObjectSnapshot(const float *array, int arrayc) {
    this->min = 0;
    this->max = 0;
    for (int i = 0; i < arrayc; i++) {
        auto value = array[i];
        this->array.emplace_back(value);
        if (i == 0 || value < this->min) this->min = value;
        if (i == 0 || value > this->max) this->max = value;
    }
    this->spread = this->max - this->min;
}

ObjectSnapshot::~ObjectSnapshot() = default;
