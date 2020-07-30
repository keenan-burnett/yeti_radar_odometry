#pragma once
#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iostream>

/*!
   \brief Enforce orthogonality conditions on the given rotation matrix such that det(R) == 1 and R.tranpose() * R = I
   \param R The input rotation matrix either 2x2 or 3x3, will be overwritten with a slightly modified matrix to
   satisfy orthogonality conditions.
*/
template<typename T>
void enforce_orthogonality(Eigen::Matrix<T, -1, -1> &R);

/*!
   \brief Retrieve the rigid transformation that transforms points in p1 into points in p2.
   The output transform type (float or double) and size SE(2) vs. SE(3) depends on the size of the input points p1, p2.
   \param p1 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param p2 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param Tf [out] This matrix will be overwritten as the output transform
   \pre p1 and p2 are the same size. p1 and p2 are the matched feature point locations between two point clouds
   \post orthogonality is enforced on the rotation matrix.
*/
template<typename T>
void get_rigid_transform(Eigen::Matrix<T, -1, -1> p1, Eigen::Matrix<T, -1, -1> p2, Eigen::Matrix<T, -1, -1> &Tf);
