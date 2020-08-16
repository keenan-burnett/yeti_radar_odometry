#include <association.hpp>

Eigen::MatrixXd cross(Eigen::VectorXd x) {
    Eigen::MatrixXd X;
    assert(x.rows() == 3 || x.rows() == 6);
    if (x.rows() == 3) {
        X = Eigen::MatrixXd::Zero(3, 3);
        X << 0, -x(2), x(1),
             x(2), 0, -x(0),
             -x(1), x(0), 0;
    } else {
        X = Eigen::MatrixXd::Zero(4, 4);
        X << 0, -x(5), x(4), x(0),
             x(5), 0, -x(3), x(1),
             -x(4), x(3), 0, x(2),
             0, 0, 0, 1;
    }
    return X;
}

// x: 4 x 1, output: 4 x 6
Eigen::MatrixXd circledot(Eigen::VectorXd x) {
    assert(x.rows() == 4);
    Eigen::Vector3d rho = x.block(0, 0, 3, 1);
    double eta = x(3);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(4, 6);
    X.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * eta;
    X.block(0, 3, 3, 3) = -1 * cross(rho);
    return X;
}

// x: 4 x 1, output: 4 x 6
Eigen::MatrixXd squaredash(Eigen::VectorXd x) {
    assert(x.rows() == 4);
    Eigen::Vector3d rho = x.block(0, 0, 3, 1);
    double eta = x(3);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(4, 6);
    X.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * eta;
    X.block(0, 3, 3, 3) = cross(rho);
    return X;
}

Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi) {
    assert(xi.rows() == 6);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector3d rho = xi.block(0, 0, 3, 1);
    Eigen::Vector3d phibar = xi.block(3, 0, 3, 1);
    double phi = phibar.norm();
    Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
    if (phi != 0) {
        phibar.normalize();
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        C = cos(phi) * I + (1 - cos(phi)) * phibar * phibar.transpose() + sin(phi) * cross(phibar);
        Eigen::Matrix3d J = Eigen::Matrix3d::Identity();
        J = I * sin(phi) / phi + (1 - sin(phi) / phi) * phibar * phibar.transpose() +
            cross(phibar) * (1 - cos(phi)) / phi;
        rho = J * rho;
    }
    T.block(0, 0, 3, 3) = C;
    T.block(0, 3, 3, 1) = rho;
    return T;
}

Eigen::MatrixXd eulerToRot(Eigen::VectorXd eul) {
    Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d C1 = C, C2 = C, C3 = C;
    C1 << 1, 0, 0, 0, cos(eul(0)), sin(eul(0)), 0, -sin(eul(0)), cos(eul(0));
    C2 << cos(eul(1)), 0, -sin(eul(1)), 0, 1, 0, sin(eul(1)), 0, cos(eul(1));
    C3 << cos(eul(2)), sin(eul(2)), 0, -sin(eul(2)), cos(eul(2)), 0, 0, 0, 1;
    C = C1 * C2 * C3;
    return C;
}

// Jacobian F = di f / di g evaluated at gbar where gbar is 4 x 1 (x,y,z,1) points in local frame
// f(.) converts (x,y,z) into cylindrical coordinates
Eigen::MatrixXd MotionDistortedRansac::get_jacobian(Eigen::Vector4d gbar) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(4, 4);
    double x = gbar(0);
    double y = gbar(1);
    J(0, 0) = x / sqrt(pow(x, 2) + pow(y, 2));
    J(0, 1) = y / sqrt(pow(x, 2) + pow(y, 2));
    J(1, 0) = -y / (pow(x, 2) + pow(y, 2));
    J(1, 1) = x / (pow(x, 2) + pow(y, 2));
    return J;
}

// Jacobian H = di h / di x evaluated at gbar where gbar is 4 x 1 (x,y,z,1) points in local frame
// h(.) converts (r,theta,z) into cartesian coordinates
Eigen::MatrixXd MotionDistortedRansac::get_inv_jacobian(Eigen::Vector4d gbar) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(4, 4);
    Eigen::VectorXd xbar = to_cylindrical(gbar);
    double r = xbar(0);
    double theta = xbar(1);
    J(0, 0) = cos(theta);
    J(0, 1) = -r * sin(theta);
    J(1, 0) = sin(theta);
    J(1, 1) = r * cos(theta);
    return J;
}

// f(.) convert (x, y, z) into cylindrical coordinates (r, theta, z)
Eigen::VectorXd MotionDistortedRansac::to_cylindrical(Eigen::VectorXd gbar) {
    Eigen::VectorXd ybar = Eigen::VectorXd::Zero(4, 1);
    double x = gbar(0);
    double y = gbar(1);
    ybar(0) = sqrt(pow(x, 2) + pow(y, 2));
    ybar(1) = atan2(y, x);
    ybar(2) = gbar(2);
    ybar(3) = 1;
    return ybar;
}

// inv(f(.)) converts (r, theta, z) into cartesian coordinates (x, y, z)
Eigen::VectorXd MotionDistortedRansac::from_cylindrical(Eigen::VectorXd ybar) {
    Eigen::VectorXd p = Eigen::VectorXd::Zero(4, 1);
    double r = ybar(0);
    double theta = ybar(1);
    p(0) = r * cos(theta);
    p(1) = r * sin(theta);
    p(2) = ybar(2);
    p(3) = 1;
    return p;
}

double MotionDistortedRansac::get_delta_t(Eigen::VectorXd y2, Eigen::VectorXd y1) {
    double azimuth2 = y2(1);
    double azimuth1 = y1(1);
    double min_diff = 10000;
    int closest_azimuth = 0;
    for (uint i = 0; i < a1.size(); ++i) {
        if (fabs(a1[i] - azimuth1) < min_diff) {
            min_diff = fabs(a1[i] - azimuth1);
            closest_azimuth = i;
        }
    }
    int64_t time1 = t1[closest_azimuth];
    min_diff = 10000;
    closest_azimuth = 0;
    for (uint i = 0; i < a2.size(); ++i) {
        if (fabs(a2[i] - azimuth2) < min_diff) {
            min_diff = fabs(a2[i] - azimuth2);
            closest_azimuth = i;
        }
    }
    int64_t time2 = t2[closest_azimuth];
    int64_t delta_t = time2 - time1;
    return double(delta_t) / 1000000.0;
}

// Pre: set wbar to zero or use the wbar from the previous iteration?
void MotionDistortedRansac::get_motion_parameters(Eigen::MatrixXd& p1small, Eigen::MatrixXd& p2small,
    std::vector<double> delta_t_local, std::vector<double> delta_theta_rad_local, Eigen::VectorXd &wbar) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
    std::vector<double> error(100, 0.0);
    for (int it = 0; it < 100; ++it) {
        for (int m = 0; m < p1small.cols(); ++m) {
            double delta_t = delta_t_local[m];
            Eigen::MatrixXd Tbar = se3ToSE3(delta_t * wbar);
            double theta = delta_theta_rad_local[m];
            // Eigen::MatrixXd C = Eigen::MatrixXd::Identity(4, 4);
            // C(0, 0) = cos(theta); C(0, 1) = sin(theta);
            // C(1, 0) = -sin(theta); C(1, 1) = cos(theta);
            Eigen::VectorXd gbar = Tbar * p1small.block(0, m, 4, 1);
            Eigen::MatrixXd G = delta_t * circledot(gbar);
            std::cout << "p1: " << p1small.block(0, m, 4, 1) << std::endl;
            std::cout << "p2: " << p2small.block(0, m, 4, 1) << std::endl;
            std::cout << "delta_t: " << delta_t << std::endl;
            std::cout << "theta: " << theta << std::endl;

            Eigen::VectorXd ebar = p2small.block(0, m, 4, 1) - gbar;
            // Eigen::MatrixXd H = get_inv_jacobian(gbar);
            // Eigen::MatrixXd R_cart = H * R_pol * H.transpose();
            // R_cart = R_cart.inverse();
            // A += G.transpose() * R_cart * G;
            // b += G.transpose() * R_cart * ebar;
            A += G.transpose() * G;
            b += G.transpose() * ebar;
        }
        Eigen::VectorXd delta_w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        // Line search for best update
        double minError = 10000000;
        double bestAlpha = 1.0;
        for (double alpha = 0.1; alpha <= 1.0; alpha += 0.1) {
            double e = 0;
            Eigen::VectorXd wbar_temp = wbar + alpha * delta_w;
            for (int m = 0; m < p1small.cols(); ++m) {
                double delta_t = delta_t_local[m];
                Eigen::MatrixXd Tbar = se3ToSE3(delta_t * wbar_temp);
                double theta = delta_theta_rad_local[m];
                // Eigen::MatrixXd C = Eigen::MatrixXd::Identity(4, 4);
                // C(0, 0) = cos(theta); C(0, 1) = sin(theta);
                // C(1, 0) = -sin(theta); C(1, 1) = cos(theta);
                Eigen::VectorXd ebar = p2small.block(0, m, 4, 1) - Tbar * p1small.block(0, m, 4, 1);
                e += ebar.norm();
            }
            if (e < minError) {
                minError = e;
                bestAlpha = alpha;
            }
        }
        wbar = wbar + bestAlpha * delta_w;
        // std::cout << "it: " << it << " error: " << minError << std::endl;
        // std::cout << "wbar: " << wbar << std::endl;
    }
}

void MotionDistortedRansac::get_motion_parameters2(Eigen::MatrixXd& p1small, Eigen::MatrixXd& p2small,
    std::vector<double> delta_t_local, std::vector<double> delta_theta_rad_local, Eigen::VectorXd &wbar) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(3, 4);
    for (int m = 0; m < p1small.cols(); ++m) {
        double delta_t = delta_t_local[m];
        Eigen::VectorXd q = P * (p2small.block(0, m, 4, 1) - p1small.block(0, m, 4, 1));
        Eigen::MatrixXd Q = P * circledot(p1small.block(0, m, 4, 1));
        A += pow(delta_t, 2) * Q.transpose() * Q;
        b += delta_t * Q.transpose() * q;
    }
    wbar = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

void MotionDistortedRansac::get_motion_parameters3(Eigen::MatrixXd& p1small, Eigen::MatrixXd& p2small,
    std::vector<double> delta_t_local, std::vector<double> delta_theta_rad_local, Eigen::VectorXd &wbar) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6);

    for (int it = 0; it < 100; ++it) {
        for (int m = 0; m < p1small.cols(); ++m) {
            double delta_t = delta_t_local[m];


            Eigen::Matrix4d Tbar = Eigen::Matrix4d::Identity();
            Eigen::VectorXd xi = delta_t * wbar;

            Eigen::Vector3d rho = xi.block(0, 0, 3, 1);
            Eigen::Vector3d phibar = xi.block(3, 0, 3, 1);
            double phi = phibar.norm();
            Eigen::Matrix3d C = Eigen::Matrix3d::Identity();
            Tbar.block(0, 3, 3, 1) = rho;
            Eigen::Vector3d r = rho;
            Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
            if (phi != 0) {
                Eigen::VectorXd abar = phibar.normalized();
                // phibar.normalize();
                Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
                // C = cos(phi) * I + (1 - cos(phi)) * phibar * phibar.transpose() + sin(phi) * cross(phibar);
                S = I * sin(phi) / phi + (1 - sin(phi) / phi) * abar * abar.transpose() -
                    cross(abar) * (1 - cos(phi)) / phi;
                r = S * rho;
                C = I - cross(phibar) * S;
            }
            Tbar.block(0, 0, 3, 3) = C;
            Tbar.block(0, 3, 3, 1) = r;
            // Eigen::MatrixXd Tbar = se3ToSE3(delta_t * wbar);
            Eigen::VectorXd gbar = Tbar * p1small.block(0, m, 4, 1);
            // Eigen::MatrixXd G = delta_t * circledot(gbar);
            Eigen::MatrixXd T_adj_inv = Eigen::MatrixXd::Zero(6, 6);
            T_adj_inv.block(0, 0, 3, 3) = C.transpose();
            T_adj_inv.block(3, 3, 3, 3) = C.transpose();
            T_adj_inv.block(0, 3, 3, 3) = C.transpose() * cross(r);
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3, 3);
            if (phi == 0) {
                Q = -0.5 * cross(rho);
            } else {
            Q = -0.5 * cross(rho) + ((phi - sin(phi)) / (pow(phi, 3))) * (cross(phibar) * cross(rho) + cross(rho) * cross(phibar) - cross(phibar) * cross(rho) * cross(phibar)) +  // NOLINT
                ((1 - pow(phi, 2) / 2 - cos(phi)) / pow(phi, 4)) * (cross(phibar) * cross(phibar) * cross(rho) + cross(rho) * cross(phibar) * cross(phibar) - 3 * cross(phibar) * cross(rho) * cross(phibar))  // NOLINT
                - 0.5 * (((1 - pow(phi, 2) / 2 - cos(phi)) / pow(phi, 4)) - 3 * (phi - sin(phi) - pow(phi, 3) / 6) / pow(phi, 5)) * (cross(phibar) * cross(rho) * cross(phibar) * cross(phibar) + cross(phibar) * cross(phibar) * cross(rho) * cross(phibar));  // NOLINT
            }
            Eigen::MatrixXd S_bar = Eigen::MatrixXd::Zero(6, 6);
            S_bar.block(0, 0, 3, 3) = S;
            S_bar.block(3, 3, 3, 3) = S;
            S_bar.block(0, 3, 3, 3) = Q;


            Eigen::MatrixXd G = delta_t * Tbar * squaredash(p1small.block(0, m, 4, 1)) * T_adj_inv * S_bar;
            Eigen::VectorXd ebar = p2small.block(0, m, 4, 1) - gbar;
            Eigen::MatrixXd H = get_inv_jacobian(gbar);
            // Eigen::MatrixXd R_cart = H * R_pol * H.transpose();
            // R_cart = R_cart.inverse();
            // A += G.transpose() * R_cart * G;
            // b += G.transpose() * R_cart * ebar;
            A += G.transpose() * G;
            b += G.transpose() * ebar;
        }
        Eigen::VectorXd delta_w = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        wbar += delta_w;
    }
}

void MotionDistortedRansac::getInliers(Eigen::VectorXd wbar, std::vector<int> &inliers) {
    for (uint i = 0; i < p1bar.cols(); ++i) {
        Eigen::VectorXd p2 = p2bar.block(0, i, 4, 1);
        Eigen::VectorXd p1 = p1bar.block(0, i, 4, 1);
        double delta_t = delta_ts[i];
        Eigen::MatrixXd Tm = se3ToSE3(delta_t * wbar);
        Eigen::VectorXd error = p2 - Tm * p1;
        if (error.norm() < tolerance)
            inliers.push_back(i);
    }
}

int MotionDistortedRansac::computeModel() {
    std::vector<int> best_inliers;
    int subset_size = 2;
    iterations = 1;
    for (int i = 0; i < iterations; ++i) {
        std::vector<int> subset = random_subset(p1bar.cols(), subset_size);
        // NLLS to obtain the motion estimate
        Eigen::MatrixXd p1small, p2small;
        p1small = Eigen::MatrixXd::Zero(4, subset_size);
        p2small = Eigen::MatrixXd::Zero(4, subset_size);
        std::vector<double> delta_t_local(subset_size, 0.0);
        std::vector<double> delta_theta_rad_local(subset_size, 0.0);
        for (int j = 0; j < subset_size; ++j) {
            p1small.block(0, j, 4, 1) = p1bar.block(0, subset[j], 4, 1);
            p2small.block(0, j, 4, 1) = p2bar.block(0, subset[j], 4, 1);
            delta_t_local[j] = delta_ts[subset[j]];
            delta_theta_rad_local[j] = delta_theta_rads[subset[j]];
        }

        Eigen::VectorXd wbar = Eigen::VectorXd::Zero(6, 1);
        get_motion_parameters(p1small, p2small, delta_t_local, delta_theta_rad_local, wbar);
        // std::cout << "wbar: " << wbar << std::endl;
        // Check the number of inliers
        std::vector<int> inliers;
        getInliers(wbar, inliers);
        // std::cout << "inliers: " << inliers.size() << std::endl;
        if (inliers.size() > best_inliers.size()) {
            best_inliers = inliers;
        }
        if (double(inliers.size()) / double(p1bar.cols()) > inlier_ratio)
            break;
    }
    // Refine transformation using the inlier set
    Eigen::MatrixXd p1small, p2small;
    p1small = Eigen::MatrixXd::Zero(4, best_inliers.size());
    p2small = p1small;
    std::vector<double> delta_t_local(best_inliers.size(), 0.0);
    std::vector<double> delta_theta_rad_local(best_inliers.size(), 0.0);
    for (uint j = 0; j < best_inliers.size(); ++j) {
        p1small.block(0, j, 4, 1) = p1bar.block(0, best_inliers[j], 4, 1);
        p2small.block(0, j, 4, 1) = p2bar.block(0, best_inliers[j], 4, 1);
        delta_t_local[j] = delta_ts[best_inliers[j]];
        delta_theta_rad_local[j] = delta_theta_rads[best_inliers[j]];
    }
    get_motion_parameters(p1small, p2small, delta_t_local, delta_theta_rad_local, w_best);
    // std::cout << "iterations: " << i << std::endl;
    std::cout << "inlier ratio: " << double(best_inliers.size()) / double(p1bar.cols()) << std::endl;
    return best_inliers.size();
}
