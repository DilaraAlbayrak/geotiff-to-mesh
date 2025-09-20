#pragma once
#include <iostream>
#include <vector>

#include "MeshGenerator.h"

int main(int argc, char* argv[])
{
    std::string inputFile;
    std::string outputFile;
    double zScale;

    auto getBaseName = [](const std::string& fileName) -> std::string {
        size_t last_dot_pos = fileName.rfind('.');
        if (last_dot_pos != std::string::npos) 
            return fileName.substr(0, last_dot_pos);

        return fileName;
        };

	// no arguments provided, use default values
    if (argc == 1)
    {
		inputFile = "NAC_DTM_APOLLO11.TIF"; // 1055.5m X 6989m, 2m/px resolution
		outputFile = getBaseName(inputFile) + ".obj"; 
        zScale = 1.5;                      
    }
    else if (argc == 3 || argc == 4)
    {
        std::cout << ">>>> running with command-line arguments.\n";
        inputFile = argv[1];
        outputFile = argv[2];
		// If a third argument is provided, use it as the zScale, otherwise default to 1.0
        zScale = (argc == 4) ? std::stod(argv[3]) : 1.0;
    }
    else
    {
        std::cerr << ">>>> usage: " << argv[0] << " <input.tif> <output.obj> [z_scale]\n";
        std::cerr << ">>>>  z_scale (optional): A factor to exaggerate the vertical terrain. Default is 1.0.\n";
        return 1; 
    }

    // Initialise GDAL
    GDALAllRegister();

    try
    {
        std::cout << ">>>> input file: " << inputFile << std::endl;
        std::cout << ">>>> output file: " << outputFile << std::endl;
        std::cout << ">>>> vertical exaggeration (zScale): " << zScale << std::endl;

        DigitalTerrainModel dtm(inputFile);
        MeshGenerator generator;
		int downscaleFactor = 2; // 1/downscaleFactor^2 resolution, 2 means that it downscaled to 1/4  
        generator.generateMesh(dtm, outputFile, downscaleFactor, zScale);
        generator.generateHeightmap(dtm, getBaseName(inputFile) + "_heightmap_8-bit.png", zScale);
        generator.generateHeightmap16bit(dtm, getBaseName(inputFile) + "_heightmap_16-bit.png", zScale);
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n>>>> an error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}