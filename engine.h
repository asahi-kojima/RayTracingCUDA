#pragma once
#include "world.h"
#include "render_target.h"

class RayTracingEngine
{
public:
    static void render(World& world, RenderTarget& renderTarget, const u32 sampleSize = 30, const u32 depth = 50);
};
