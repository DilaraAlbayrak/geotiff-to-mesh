#pragma once
#include "DigitalTerrainModel.h"

class MeshGenerator
{
public:
	void generateMesh(const DigitalTerrainModel& dtm, const std::string& outputFilePath, int downscaleFactor = 1, double zScale = 1.0) const;
    void generateHeightmap(const DigitalTerrainModel& dtm, const std::string& outputFilePath, double zScale=1.0, int downscaleFactor=1) const;
	void generateHeightmap16bit(const DigitalTerrainModel& dtm, const std::string& outputFilePath, double zScale=1.0, int downscaleFactor = 1) const;
};

