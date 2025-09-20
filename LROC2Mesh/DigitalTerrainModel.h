#pragma once
#include <string>
#include <gdal_priv.h>

constexpr float DEFAULT_NO_DATA_VALUE = -3.40282266e+38f; // -3.40282266e+38f for APOLLO 11 DTM 

class DigitalTerrainModel
{
private:
    // Make the class non-copyable to prevent issues with pointer ownership.
    DigitalTerrainModel(const DigitalTerrainModel&) = delete;
    DigitalTerrainModel& operator=(const DigitalTerrainModel&) = delete;

    std::string _filePath;
    GDALDataset* _dataset;
    GDALRasterBand* _rasterBand;
    int _width;
    int _height;

	// GeoTransform parameters for the raster dataset
	// 0 : top left x, 
	// 1 : pixel width, 
	// 2 : rotation (0 if North is up), 
	// 3 : top left y, 4 : rotation (0 if North is up), 
	// 5 : pixel height (negative value for North up)
	double _geoTransform[6] = { 0.0, 1.0, 0.0, 0.0, 0.0, -1.0 }; // Default values
    float _noDataValue;

public:
	// explicit constructor because we want to avoid implicit conversions
	explicit DigitalTerrainModel(const std::string& filePath) : _filePath(filePath), _dataset(nullptr), _rasterBand(nullptr), _width(0), _height(0), _noDataValue(0.0f)
	{
		_dataset = (GDALDataset*)GDALOpen(filePath.c_str(), GA_ReadOnly);
		if (!_dataset)
		{
			throw std::runtime_error(">>>> failed to open the DTM file: " + filePath);
		}

		if (_dataset->GetGeoTransform(_geoTransform) != CE_None)
		{
			throw std::runtime_error(">>>> failed to get GeoTransform from file: " + _filePath);
		}

		_rasterBand = _dataset->GetRasterBand(1);
		if (!_rasterBand)
		{
			throw std::runtime_error(">>>> failed to get raster band 1 from file: " + _filePath);
		}

		_width = _rasterBand->GetXSize();
		_height = _rasterBand->GetYSize();

		int hasNoData;
		_noDataValue = _rasterBand->GetNoDataValue(&hasNoData);
		if (!hasNoData)
		{
			_noDataValue = DEFAULT_NO_DATA_VALUE; // Use default if no NoData value is set
		}
	}
	// Destructor to clean up GDAL resources
	~DigitalTerrainModel()
	{
		if (_dataset)
		{
			GDALClose(_dataset);
			_dataset = nullptr; // probably not necessary, but good practice
		}
	}

	int getWidth() const { return _width; }
	int getHeight() const { return _height; }
	float getNoDataValue() const { return _noDataValue; }
	const double* getGeoTransform() const { return _geoTransform; }

	bool readScanline(int y, std::vector<float>& buffer) const
	{
		if (y < 0 || y >= _height)
			return false; // Invalid scanline index

		buffer.resize(_width);
		// Read a single scanline (row) of data from the raster band
		// - GF_Read: operation type (read)
		// - 0: x offset (start at the left edge)
		// - y: y offset (row index)
		// - _width: number of pixels to read in the x direction
		// - 1: number of rows to read (1 row)
		// - buffer.data(): pointer to the output buffer where the data will be stored
		// - _width: width of the output buffer
		// - 1: height of the output buffer (1 row)
		// - GDT_Float32: data type of the output buffer
		// - 0: pixel space (no pixel offset)
		// - 0: line space (no line offset)
		CPLErr result = _rasterBand->RasterIO(GF_Read, 0, y, _width, 1, buffer.data(), _width, 1, GDT_Float32, 0, 0);

		return result == CE_None; 
	}
};