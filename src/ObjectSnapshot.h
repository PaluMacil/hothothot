// ObjectSnapshot.h
// Dan Wolf

#ifndef HOTHOTHOT_OBJECTSNAPSHOT_H
#define HOTHOTHOT_OBJECTSNAPSHOT_H

#include <vector>

class ObjectSnapshot {
public:
    std::vector<float> array;
    float min{};
    float max{};
    float spread{};

    explicit ObjectSnapshot(const float* array, int arrayc);
    explicit ObjectSnapshot();

    ~ObjectSnapshot();
};


#endif //HOTHOTHOT_OBJECTSNAPSHOT_H
