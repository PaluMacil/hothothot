// ObjectSnapshot_test.cpp
// Dan Wolf

#include "gtest/gtest.h"
#include "ObjectSnapshot.h"

TEST(ObjectSnapshot, Spread) {
    float arr[] = {4.4, 6, 20, 1, 7, 3, 9, 14};
    ObjectSnapshot snap(arr, 8);

    EXPECT_FLOAT_EQ(snap.spread, 19);
}

TEST(ObjectSnapshot, MinMax) {
    float arr[] = {4.4, 6, 20, 1, 7, 3, 9, 14};
    ObjectSnapshot snap(arr, 8);

    EXPECT_FLOAT_EQ(snap.min, 1);
    EXPECT_FLOAT_EQ(snap.max, 20);
}