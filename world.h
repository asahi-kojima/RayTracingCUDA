#pragma once
#include <vector>
#include "hittable.h"

class World
{
public:
    void addObject();

    void buildWorldOnGpu();

private:
    std::vector<Hittable*> mHittableList;
};