#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "association.hpp"

int main() {
    // Generate fake data using a known motion distortion
    float v = 5.0;
    float omega = 10.0;

    Eigen::MatrixXf p1 = Eigen::MatrixXf::Random(2, N);

    // Simulate the generation of two clouds, motion-distorted:
    



    // (Add noise)

    // run the motion-distorted RANSAC to extract the desired transform, motion parameters?

    return 0;
}
