#include <association.hpp>

Eigen::MatrixXf cross(Eigen::VectorXf x) {
    Eigen::MatrixXf X;
    assert(x.rows() == 3 || x.rows() == 6);
    if (x.rows() == 3) {
        X = Eigen::MatrixXf::Zero(3, 3);
        X << 0, -x[2], x[1],
             x[2], 0, -x[0],
             -x[1], x[0], 0;
    } else {
        X = Eigen::MatrixXf::Zero(4, 4);
        X << 0, -x[5], x[4], x[0],
             x[5], 0, -x[3], x[1],
             -x[4], x[3], 0, x[2],
             0, 0, 0, 1;
    }
    return X;
}

// x: 4 x 1, output: 4 x 6
Eigen::MatrixXf circledot(Eigen::VectorXf x) {
    assert(x.rows() == 4);
    Eigen::Vector3f rho = x.block(0, 0, 3, 1);
    float eta = x(3);
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(4, 6);
    X.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity() * eta;
    X.block(0, 3, 3, 3) = -1 * cross(rho);
    return X;
}

Eigen::Matrix4f se3ToSE3(Eigen::MatrixXf xi) {
    assert(xi.rows() == 6);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Vector3f rho = xi.block(0, 0, 3, 1);
    Eigen::Vector3f phibar = xi.block(0, 3, 3, 1);
    float phi = phibar.norm();
    Eigen::Matrix3f C = Eigen::Matrix3f::Identity();
    if (phi != 0) {
        phibar.normalize();
        Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
        C = cos(phi) * I + (1 - cos(phi)) * phibar * phibar.transpose() + sin(phi) * cross(phibar);
        Eigen::Matrix3f J = Eigen::Matrix3f::Identity();
        J = I * sin(phi) / phi + (1 - sin(phi) / phi) * phibar * phibar.transpose() +
            cross(phibar) * (1 - cos(phi)) / phi;
        rho = J * rho;
    }
    T.block(0, 0, 3, 3) = C;
    T.block(0, 3, 3, 1) = rho;
    return T;
}

// Jacobian F = di f / di g evaluated at gbar where gbar is 3 x 1 (x,y,z) points in local frame
// f(.) converts (x,y,z) into cylindrical coordinates
Eigen::MatrixXf MotionDistortedRansac::get_jacobian(Eigen::Vector3f gbar) {
    Eigen::MatrixXf J = Eigen::MatrixXf::Identity(4, 4);
    float x = gbar(0);
    float y = gbar(1);
    J(0, 0) = x / sqrt(pow(x, 2) + pow(y, 2));
    J(0, 1) = y / sqrt(pow(x, 2) + pow(y, 2));
    J(1, 0) = -y / (pow(x, 2) + pow(y, 2));
    J(1, 1) = x / (pow(x, 2) + pow(y, 2));
    return J;
}

// Jacobian H = di h / di x evaluated at gbar where gbar is 3 x 1 (x,y,z) points in local frame
// h(.) converts (r,theta,z) into cartesian coordinates
Eigen::MatrixXf MotionDistortedRansac::get_inv_jacobian(Eigen::Vector3f gbar) {
    Eigen::MatrixXf J = Eigen::MatrixXf::Identity(4, 4);
    Eigen::VectorXf xbar = to_cylindrical(gbar);
    float r = xbar(0);
    float theta = xbar(1);
    J(0, 0) = cos(theta);
    J(0, 1) = -r * sin(theta);
    J(1, 0) = sin(theta);
    J(1, 1) = r * cos(theta);
    return J;
}

// f(.) convert (x, y, z) into cylindrical coordinates (r, theta, z)
Eigen::VectorXf MotionDistortedRansac::to_cylindrical(Eigen::VectorXf gbar) {
    Eigen::VectorXf ybar = Eigen::VectorXf::Zero(4, 1);
    float x = gbar(0);
    float y = gbar(1);
    ybar(0) = sqrt(pow(x, 2) + pow(y, 2));
    ybar(1) = atan2(y, x);
    ybar(2) = gbar(2);
    ybar(3) = 1;
    return ybar;
}

// inv(f(.)) converts (r, theta, z) into cartesian coordinates (x, y, z)
Eigen::VectorXf MotionDistortedRansac::from_cylindrical(Eigen::VectorXf ybar) {
    Eigen::VectorXf p = Eigen::VectorXf::Zero(4, 1);
    float r = ybar(0);
    float theta = ybar(1);
    p(0) = r * cos(theta);
    p(1) = r * sin(theta);
    p(2) = ybar(2);
    p(3) = 1;
    return p;
}

float MotionDistortedRansac::get_delta_t(Eigen::VectorXf y2, Eigen::VectorXf y1) {
    float azimuth2 = y2(1);
    float azimuth1 = y1(1);
    float min_diff = 10000;
    int closest_azimuth = 0;
    for (uint i = 0; i < a1.size(); ++i) {
        if (fabs(a1[i] - azimuth1) < min_diff) {
            min_diff = fabs(a1[i] = azimuth1);
            closest_azimuth = i;
        }
    }
    int64_t time1 = t1[closest_azimuth];
    min_diff = 10000;
    closest_azimuth = 0;
    for (uint i = 0; i < a2.size(); ++i) {
        if (fabs(a2[i] - azimuth2) < min_diff) {
            min_diff = fabs(a2[i] = azimuth2);
            closest_azimuth = i;
        }
    }
    int64_t time2 = t2[closest_azimuth];
    int64_t delta_t = time2 - time1;
    return float(delta_t) / 1000000.0;
}

// Pre: set wbar to zero or use the wbar from the previous iteration?
void MotionDistortedRansac::get_motion_parameters(Eigen::MatrixXf& p1small, Eigen::MatrixXf& p2small,
    Eigen::VectorXf &wbar) {
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(6);
    wbar = Eigen::VectorXf::Zero(6);
    for (int it = 0; it < max_gn_iterations; ++it) {
        for (int m = 0; m < p1small.cols(); ++m) {
            Eigen::VectorXf p2 = p2small.block(0, m, 4, 1);
            Eigen::VectorXf y2 = to_cylindrical(p2small.block(0, m, 4, 1));
            Eigen::VectorXf p1 = p1small.block(0, m, 4, 1);
            Eigen::VectorXf y1 = to_cylindrical(p1);
            float delta_t = get_delta_t(y2, y1);

            Eigen::MatrixXf Tbar = se3ToSE3(delta_t * cross(wbar));
            Eigen::MatrixXf G = delta_t * circledot(Tbar * p1);
            Eigen::VectorXf gbar = Tbar * p1;

            Eigen::VectorXf ebar = p2 - gbar;
            Eigen::MatrixXf H = get_inv_jacobian(gbar);
            Eigen::MatrixXf R_cart = H * R_pol * H.transpose();
            R_cart = R_cart.inverse();
            A += G.transpose() * R_cart * G;
            b += G.transpose() * R_cart * ebar;
        }
        Eigen::VectorXf delta_w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        wbar += delta_w;
        // TODO(keenan): Check for convergence
    }
}

void MotionDistortedRansac::getInliers(Eigen::VectorXf wbar, std::vector<int> &inliers) {
    for (uint i = 0; i < p1bar.cols(); ++i) {
        Eigen::VectorXf p2 = p2bar.block(0, i, 4, 1);
        Eigen::VectorXf p1 = p1bar.block(0, i, 4, 1);
        Eigen::VectorXf y2 = to_cylindrical(p2);
        Eigen::VectorXf y1 = to_cylindrical(p1);
        float delta_t = get_delta_t(y2, y1);
        Eigen::MatrixXf Tm = se3ToSE3(delta_t * cross(wbar));
        Eigen::VectorXf error = p2 - Tm * p1;
        if (error.norm() < tolerance)
            inliers.push_back(i);
    }
}

int MotionDistortedRansac::computeModel() {
    std::vector<int> best_inliers;
    for (int i = 0; i < iterations; ++i) {
        std::vector<int> subset = random_subset(p1bar.cols(), dim);
        // NLLS to obtain the motion estimate
        Eigen::MatrixXf p1small, p2small;
        p1small = Eigen::MatrixXf::Zero(4, dim);
        p2small = p1small;
        for (int j = 0; j < dim; ++j) {
            p1small.block(0, j, 4, 1) = p1bar.block(0, subset[j], 4, 1);
            p2small.block(0, j, 4, 1) = p2bar.block(0, subset[j], 4, 1);
        }
        Eigen::VectorXf wbar = Eigen::VectorXf::Zero(6, 1);
        get_motion_parameters(p1small, p2small, wbar);
        // Check the number of inliers
        std::vector<int> inliers;
        getInliers(wbar, inliers);
        if (inliers.size() > best_inliers.size()) {
            best_inliers = inliers;
        }
        if (float(inliers.size()) / float(p1bar.cols()) > inlier_ratio)
            break;
    }
    // Refine transformation using the inlier set
    Eigen::MatrixXf p1small, p2small;
    p1small = Eigen::MatrixXf::Zero(4, best_inliers.size());
    p2small = p1small;
    for (uint j = 0; j < best_inliers.size(); ++j) {
        p1small.block(0, j, 4, 1) = p1bar.block(0, best_inliers[j], 4, 1);
        p2small.block(0, j, 4, 1) = p2bar.block(0, best_inliers[j], 4, 1);
    }
    get_motion_parameters(p1small, p2small, w_best);
    // std::cout << "iterations: " << i << std::endl;
    // std::cout << "inlier ratio: " << float(max_inliers) / float(p1.cols()) << std::endl;
    return best_inliers.size();
}
