#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <string>
// #include <boost/shared_ptr.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/version.hpp>
#include <opencv2/core.hpp>


// namespace fs = boost::filesystem;

int main() {
    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    // float cart_resolution = 0.25;
    // int cart_pixel_width = 1000;
    // bool interpolate_crossover = true;


    DIR *dirp = opendir(datadir.c_str());
    struct dirent *dp;
    std::vector<std::string> radar_files;
    while ((dp = readdir(dirp)) != NULL) {
        radar_files.push_back(dp->d_name);
    }

    std::cout << "radar files: " << radar_files.size() << std::endl;

    float radar_resolution = 0.0432;
    std::vector<int64_t> timestamps;
    std::vector<float> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;
    


    return 0;
}
