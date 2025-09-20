#include "MeshGenerator.h"
#include <iostream>
#include <fstream>

// for openmp
#include <sstream> 
#include <omp.h>  

#include <chrono>
#include <iomanip> // Required for std::setprecision

// for heightmap generation
#include <algorithm> // for std::min_element, std::max_element
#include <cmath> // for std::round
#include <algorithm>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "lodepng.h"

struct Vector3
{
	double x = 0.0, y = 0.0, z = 0.0;

	Vector3& operator+=(const Vector3& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	void normalize()
	{
		double length = std::sqrt(x * x + y * y + z * z);
		if (length > 1e-6) { // avoid division by zero
			x /= length;
			y /= length;
			z /= length;
		}
	}
};

Vector3 cross(const Vector3& a, const Vector3& b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// Overload the - operator for Vector3
Vector3 operator-(const Vector3& a, const Vector3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

void MeshGenerator::generateMesh(const DigitalTerrainModel& dtm, const std::string& outputFilePath, int downscaleFactor, double zScale) const
{
    if (downscaleFactor < 1) {
        std::cout << ">>>> warning: downscaleFactor cannot be less than 1. Setting to 1 (no downscaling).\n";
        downscaleFactor = 1;
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << ">>>> generating mesh for DTM with normals (Downscale: " << downscaleFactor << "x, OpenMP: enabled)" << std::endl;

    const int width = dtm.getWidth();
    const int height = dtm.getHeight();

    // Generate vertex data in memory
    std::cout << "\n>>>> [1/5] generating vertices in memory...\n";
    std::vector<Vector3> vertices;
    std::vector<int> vertexMap(static_cast<size_t>(width) * height, 0);

    const double* geoTransform = dtm.getGeoTransform();
    const float noDataValue = dtm.getNoDataValue();
    std::vector<float> scanlineBuffer(width);

    for (int y = 0; y < height; y += downscaleFactor) {
        if (!dtm.readScanline(y, scanlineBuffer)) continue;
        for (int x = 0; x < width; x += downscaleFactor) {
            float elevation = scanlineBuffer[x];
            if (elevation == noDataValue) continue;

            Vector3 pos;
            pos.x = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2];
            pos.y = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5];
            pos.z = -elevation * zScale;

            vertices.push_back(pos);
            vertexMap[static_cast<size_t>(y) * width + x] = static_cast<int>(vertices.size());
        }
    }
    std::cout << ">>>> generated " << vertices.size() << " vertices.\n";

    // Center the Model to Origin
    if (!vertices.empty()) {
        std::cout << "\n>>>> [2/5] centering the model to origin...\n";
        Vector3 min_bound = vertices[0];
        Vector3 max_bound = vertices[0];

        for (size_t i = 1; i < vertices.size(); ++i) {
            min_bound.x = std::min(min_bound.x, vertices[i].x);
            min_bound.y = std::min(min_bound.y, vertices[i].y);
            min_bound.z = std::min(min_bound.z, vertices[i].z);
            max_bound.x = std::max(max_bound.x, vertices[i].x);
            max_bound.y = std::max(max_bound.y, vertices[i].y);
            max_bound.z = std::max(max_bound.z, vertices[i].z);
        }

        Vector3 center;
        center.x = min_bound.x + (max_bound.x - min_bound.x) / 2.0;
        center.y = min_bound.y + (max_bound.y - min_bound.y) / 2.0;
        center.z = min_bound.z + (max_bound.z - min_bound.z) / 2.0;

        for (auto& v : vertices) {
            v.x -= center.x;
            v.y -= center.y;
            v.z -= center.z;
        }
        std::cout << ">>>> model centered.\n";
    }

    // Calculate vertex normals
    std::cout << "\n>>>> [3/5] calculating vertex normals (in parallel)...\n";
    std::vector<Vector3> normals(vertices.size(), { 0.0, 0.0, 0.0 });

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int y = 0; y < height - downscaleFactor; y += downscaleFactor) {
            for (int x = 0; x < width - downscaleFactor; x += downscaleFactor) {
              
                size_t idx_tl = static_cast<size_t>(y) * width + x;
                size_t idx_tr = static_cast<size_t>(y) * width + (x + downscaleFactor);
                size_t idx_bl = (static_cast<size_t>(y) + downscaleFactor) * width + x;
                size_t idx_br = (static_cast<size_t>(y) + downscaleFactor) * width + (x + downscaleFactor);

                int v1_idx = vertexMap[idx_tl];
                int v2_idx = vertexMap[idx_tr];
                int v3_idx = vertexMap[idx_bl];
                int v4_idx = vertexMap[idx_br];

                if (v1_idx == 0 || v2_idx == 0 || v3_idx == 0 || v4_idx == 0) continue;

                const auto& p1 = vertices[v1_idx - 1];
                const auto& p2 = vertices[v2_idx - 1];
                const auto& p3 = vertices[v3_idx - 1];
                const auto& p4 = vertices[v4_idx - 1];

                Vector3 faceNormal1 = cross(p3 - p1, p4 - p1);

#pragma omp atomic
                normals[v1_idx - 1].x += faceNormal1.x;
#pragma omp atomic
                normals[v1_idx - 1].y += faceNormal1.y;
#pragma omp atomic
                normals[v1_idx - 1].z += faceNormal1.z;

#pragma omp atomic
                normals[v3_idx - 1].x += faceNormal1.x;
#pragma omp atomic
                normals[v3_idx - 1].y += faceNormal1.y;
#pragma omp atomic
                normals[v3_idx - 1].z += faceNormal1.z;

#pragma omp atomic
                normals[v4_idx - 1].x += faceNormal1.x;
#pragma omp atomic
                normals[v4_idx - 1].y += faceNormal1.y;
#pragma omp atomic
                normals[v4_idx - 1].z += faceNormal1.z;

                Vector3 faceNormal2 = cross(p4 - p1, p2 - p1);
#pragma omp atomic
                normals[v1_idx - 1].x += faceNormal2.x;
#pragma omp atomic
                normals[v1_idx - 1].y += faceNormal2.y;
#pragma omp atomic
                normals[v1_idx - 1].z += faceNormal2.z;

#pragma omp atomic
                normals[v4_idx - 1].x += faceNormal2.x;
#pragma omp atomic
                normals[v4_idx - 1].y += faceNormal2.y;
#pragma omp atomic
                normals[v4_idx - 1].z += faceNormal2.z;

#pragma omp atomic
                normals[v2_idx - 1].x += faceNormal2.x;
#pragma omp atomic
                normals[v2_idx - 1].y += faceNormal2.y;
#pragma omp atomic
                normals[v2_idx - 1].z += faceNormal2.z;
            }
        }

#pragma omp for
        for (int i = 0; i < normals.size(); ++i) {
            normals[i].normalize();
        }
    }
    std::cout << ">>>> normals calculated and normalized.\n";

    // Write all data to OBJ file
    std::cout << "\n>>>> [4/5] writing data to .obj file...\n";
    std::ofstream file(outputFilePath);
    if (!file.is_open()) {
        throw std::runtime_error(">>>> failed to open output file: " + outputFilePath);
    }
    file.imbue(std::locale::classic());
    file << std::fixed << std::setprecision(6);

    file << "# OBJ file generated by DTM-to-Mesh converter\n";
    file << "o DTM_Mesh\n"; 

    file << "# Vertices: " << vertices.size() << "\n";
    for (const auto& v : vertices) {
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }

    file << "\n# Vertex Normals: " << normals.size() << "\n";
    for (const auto& vn : normals) {
        file << "vn " << vn.x << " " << vn.y << " " << vn.z << "\n";
    }

    file << "\ns off\n"; 

    // Write faces with normal indices
    std::cout << "\n>>>> [5/5] writing faces with normal data...\n";
    long long faceCount = 0;
    file << "\n# Faces\n";
    for (int y = 0; y < height - downscaleFactor; y += downscaleFactor) {
        for (int x = 0; x < width - downscaleFactor; x += downscaleFactor) {
            int v1 = vertexMap[static_cast<size_t>(y) * width + x];
            int v2 = vertexMap[static_cast<size_t>(y) * width + (x + downscaleFactor)];
            int v3 = vertexMap[(static_cast<size_t>(y) + downscaleFactor) * width + x];
            int v4 = vertexMap[(static_cast<size_t>(y) + downscaleFactor) * width + (x + downscaleFactor)];

            if (v1 == 0 || v2 == 0 || v3 == 0 || v4 == 0) continue;

            file << "f " << v1 << "//" << v1 << " " << v3 << "//" << v3 << " " << v4 << "//" << v4 << "\n";
            file << "f " << v1 << "//" << v1 << " " << v4 << "//" << v4 << " " << v2 << "//" << v2 << "\n";
            faceCount += 2;
        }
    }
    file << "\n# Total faces: " << faceCount << "\n";
    std::cout << ">>>> written " << faceCount << " faces.\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    std::cout << "\n>>>> successfully generated mesh: " << outputFilePath << " in " << duration.count() << "s.\n";
}

// generates 8 bit heightmap
void MeshGenerator::generateHeightmap(const DigitalTerrainModel& dtm, const std::string& outputFilePath, double zScale, int downscaleFactor) const
{
    std::cout << ">>>> generating 8-bit PNG displacement heightmap..." << std::endl;

    const int width = dtm.getWidth();
    const int height = dtm.getHeight();
    const float noDataValue = dtm.getNoDataValue();

    // Find the exact same center_z that generateMesh uses.
    double min_z_processed = std::numeric_limits<double>::max();
    double max_z_processed = std::numeric_limits<double>::lowest();
    std::vector<float> scanlineBuffer;

    for (int y = 0; y < height; y += downscaleFactor) {
        if (!dtm.readScanline(y, scanlineBuffer)) continue;
        for (int x = 0; x < width; x += downscaleFactor) {
            const float& elevation = scanlineBuffer[x];
            if (elevation != noDataValue) {
                double processed_z = -static_cast<double>(elevation) * zScale;
                if (processed_z < min_z_processed) min_z_processed = processed_z;
                if (processed_z > max_z_processed) max_z_processed = processed_z;
            }
        }
    }
    const double center_z = min_z_processed + (max_z_processed - min_z_processed) / 2.0;

    // Read ALL pixel data for the full-resolution heightmap.
    std::vector<float> all_elevation_data;
    all_elevation_data.reserve(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        if (!dtm.readScanline(y, scanlineBuffer)) {
            scanlineBuffer.assign(width, noDataValue);
        }
        all_elevation_data.insert(all_elevation_data.end(), scanlineBuffer.begin(), scanlineBuffer.end());
    }

    // Create final displacement data for ALL pixels.
    std::vector<float> final_displacement_data(all_elevation_data.size());
    double min_disp = std::numeric_limits<double>::max();
    double max_disp = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < all_elevation_data.size(); ++i) {
        double displacement;
        if (all_elevation_data[i] == noDataValue) {
            displacement = min_z_processed - center_z;
        }
        else {
            displacement = (-static_cast<double>(all_elevation_data[i]) * zScale) - center_z;
        }
        final_displacement_data[i] = static_cast<float>(displacement);
        if (displacement < min_disp) min_disp = displacement;
        if (displacement > max_disp) max_disp = displacement;
    }
    std::cout << ">>>> final displacement range: " << min_disp << " to " << max_disp << std::endl;

    // Map the final displacement values to the 8-bit range [0, 255].
    std::vector<unsigned char> image_buffer_8bit(final_displacement_data.size());
    const double displacement_range = max_disp - min_disp;

    for (size_t i = 0; i < final_displacement_data.size(); ++i) {
        if (displacement_range > 1e-9) {
            double normalized = (final_displacement_data[i] - min_disp) / displacement_range;
            image_buffer_8bit[i] = static_cast<unsigned char>(std::round(normalized * 255.0));
        }
        else {
            image_buffer_8bit[i] = 0;
        }
    }

    auto result = std::minmax_element(image_buffer_8bit.begin(), image_buffer_8bit.end());
    std::cout << ">>>> 8-bit pixel value range: "
        << static_cast<int>(*result.first)  // Min
        << " to "
        << static_cast<int>(*result.second) // Max 
        << std::endl;

    // Save the data as an 8-bit single-channel PNG file.
    const int components = 1;
    const int stride_in_bytes = width * sizeof(unsigned char);

    if (stbi_write_png(outputFilePath.c_str(), width, height, components, image_buffer_8bit.data(), stride_in_bytes)) {
        std::cout << ">>>> successfully generated 8-bit PNG heightmap: " << outputFilePath << std::endl;
    }
    else {
        throw std::runtime_error(">>>> failed to write 8-bit PNG heightmap file.");
    }
}

void MeshGenerator::generateHeightmap16bit(const DigitalTerrainModel& dtm, const std::string& outputFilePath, double zScale, int downscaleFactor) const
{
    std::cout << ">>>> Generating 16-bit PNG displacement heightmap" << std::endl;

    const int width = dtm.getWidth();
    const int height = dtm.getHeight();
    const float noDataValue = dtm.getNoDataValue();

    // Step 1: Find center_z
    double min_z_processed = std::numeric_limits<double>::max();
    double max_z_processed = std::numeric_limits<double>::lowest();
    std::vector<float> scanlineBuffer;

    for (int y = 0; y < height; y += downscaleFactor) {
        if (!dtm.readScanline(y, scanlineBuffer)) continue;
        for (int x = 0; x < width; x += downscaleFactor) {
            const float& elevation = scanlineBuffer[x];
            if (elevation != noDataValue) {
                // Negative sign is added back to invert the elevation data, it's specific to GeoTiff used
                double processed_z = -static_cast<double>(elevation) * zScale;
                if (processed_z < min_z_processed) min_z_processed = processed_z;
                if (processed_z > max_z_processed) max_z_processed = processed_z;
            }
        }
    }
    const double center_z = min_z_processed + (max_z_processed - min_z_processed) / 2.0;

    // Read all pixel data
    std::vector<float> all_elevation_data;
    all_elevation_data.reserve(static_cast<size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        if (!dtm.readScanline(y, scanlineBuffer)) {
            scanlineBuffer.assign(width, noDataValue);
        }
        all_elevation_data.insert(all_elevation_data.end(), scanlineBuffer.begin(), scanlineBuffer.end());
    }

    // Create final displacement data
    std::vector<float> final_displacement_data(all_elevation_data.size());
    double min_disp = std::numeric_limits<double>::max();
    double max_disp = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < all_elevation_data.size(); ++i) {
        double displacement;
        if (all_elevation_data[i] == noDataValue) {
            displacement = min_z_processed - center_z;
        }
        else {
            // Negative sign is added back to invert the elevation data.
            displacement = (-static_cast<double>(all_elevation_data[i]) * zScale) - center_z;
        }
        final_displacement_data[i] = static_cast<float>(displacement);
        if (displacement < min_disp) min_disp = displacement;
        if (displacement > max_disp) max_disp = displacement;
    }
    std::cout << ">>>> Final displacement range: " << min_disp << " to " << max_disp << std::endl;

    // Normalize displacement data
    std::cout << ">>>> Normalizing displacement data to full 16-bit range [0, 65535]." << std::endl;
    std::vector<uint16_t> image_buffer_16bit(final_displacement_data.size());
    const double displacement_range = max_disp - min_disp;
    if (displacement_range < 1e-9) {
        std::fill(image_buffer_16bit.begin(), image_buffer_16bit.end(), 32767);
    }
    else {
        for (size_t i = 0; i < final_displacement_data.size(); ++i) {
            double normalized_value = (static_cast<double>(final_displacement_data[i]) - min_disp) / displacement_range * 65535.0;
            image_buffer_16bit[i] = static_cast<uint16_t>(std::round(normalized_value));
        }
    }

    auto result = std::minmax_element(image_buffer_16bit.begin(), image_buffer_16bit.end());
    std::cout << ">>>> Final 16-bit pixel value range (in memory): "
        << static_cast<int>(*result.first)
        << " to "
        << static_cast<int>(*result.second)
        << std::endl;

    // Save data using LodePNG
    std::cout << ">>>> Preparing data for LodePNG and saving..." << std::endl;
    std::vector<unsigned char> png_buffer(static_cast<size_t>(width) * height * 2);
    for (size_t i = 0; i < image_buffer_16bit.size(); ++i) {
        const uint16_t pixel_value = image_buffer_16bit[i];
        png_buffer[i * 2 + 0] = (pixel_value >> 8) & 0xFF;
        png_buffer[i * 2 + 1] = pixel_value & 0xFF;
    }

    std::vector<unsigned char> output_buffer;
    unsigned error = lodepng::encode(output_buffer, png_buffer, width, height, LCT_GREY, 16);
    if (error) { throw std::runtime_error("LodePNG encoding error " + std::to_string(error) + ": " + lodepng_error_text(error)); }

    error = lodepng::save_file(output_buffer, outputFilePath);
    if (error) { throw std::runtime_error("LodePNG file saving error " + std::to_string(error) + ": " + lodepng_error_text(error)); }

    std::cout << ">>>> Successfully generated 16-bit PNG heightmap: " << outputFilePath << std::endl;
}