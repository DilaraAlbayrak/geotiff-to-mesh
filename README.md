# Converting GeoTIFF to an .obj mesh and obtaining a heightmap
This project processes Digital Terrain Model (DTM) data from GeoTIFF files, specifically tailored for the high-resolution lunar surface data provided by NASA's Lunar Reconnaissance Orbiter Camera (LROC). It converts this geospatial raster data into render-ready assets: a high-fidelity 3D mesh in Wavefront OBJ format and high-precision heightmaps in PNG format.
The primary goal is to create assets suitable for real-time graphics applications, such as virtual reality environments or scientific visualisations, that require accurate and detailed terrain geometry.

- GeoTIFF DTM Parsing: Reads elevation, dimension, and geo-referencing data from GeoTIFF files using the GDAL library.

- High-Fidelity Mesh Generation: Creates a detailed 3D mesh (.obj) from the elevation data.

- Vertex Normal Calculation: Computes smooth per-vertex normals for realistic lighting and shading. This process is parallelised using OpenMP for efficiency.

- Model Centring: Automatically centres the generated mesh at the origin (0,0,0) for easy integration into rendering engines.

- Heightmap Generation:

    - Produces an 8-bit grayscale PNG heightmap using stb_image_write.

    - Produces a high-precision 16-bit grayscale PNG heightmap using LodePNG, preserving the vertical detail required for advanced rendering techniques like hardware tessellation.

- Allows mesh downscaling and vertical exaggeration (z-scale) via command-line arguments.

  <img width="1044" height="296" alt="data UML" src="https://github.com/user-attachments/assets/96fdd5a1-61f4-4716-a4d2-58545d23e18f" />

## Dependencies
A C++ compiler that supports C++17 and OpenMP.
GDAL (Geospatial Data Abstraction Library): This is the core dependency for reading GeoTIFF files. You must have the GDAL headers and library files installed on your system.

## References
- lodepng (for 16-bit height map generation) https://github.com/lvandeve/lodepng
- stb_image_write https://github.com/nothings/stb
- https://science.nasa.gov/resource/apollo-11-landing-site/
- The processed GeoTIFF was taken from this website https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_DTM_APOLLO11 which may not be available currently 
