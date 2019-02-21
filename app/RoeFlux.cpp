#include <iostream>

#include <cstdlib>
#include <cmath>

#include <armadillo>

#include "Conv2D/Mesh.hpp"
#include "Conv2D/EulerDefaultBase.hpp"

double uniform(double a, double b)
{
    return a + ((b - a) * std::rand() / RAND_MAX);
}

double consistencyTestError(double gamma, const EulerDefaultBase &problem)
{
    // construct random unit normal
    const double nx = uniform(-1.0, 1.0);
    const double ny = uniform(-1.0, 1.0);
    const double nl = std::hypot(nx, ny);
    const arma::rowvec n {nx / nl, ny / nl};

    // construct a state
    const double rho = uniform( 0.0, 1.0);
    const double u = uniform(-1.0, 1.0);
    const double v = uniform(-1.0, 1.0);
    const double e = uniform( 0.0, 1.0);
    const double E = e + 0.5 * (u * u + v * v);
    const arma::vec U {rho, rho * u, rho * v, rho * E};

    // compute analytical flux
    const double p = (gamma - 1.0) * rho * e;
    const double H = E + p / rho;
    const arma::vec F {
            n[0] * (rho * u)         + n[1] * (rho * v),
            n[0] * (rho * u * u + p) + n[1] * (rho * u * v),
            n[0] * (rho * u * v)     + n[1] * (rho * v * v + p),
            n[0] * (rho * u * H)     + n[1] * (rho * v * H)};

    // compute Roe flux
    arma::vec FHat(4);
    double s;
    problem.computeRoeFlux(U, U, n, FHat, s);

    // return error
    return arma::norm(F - FHat);
}

double flipTestError(const EulerDefaultBase &problem)
{
    // construct random oppsing unit normals
    const double nx = uniform(-1.0, 1.0);
    const double ny = uniform(-1.0, 1.0);
    const double nl = std::hypot(nx, ny);
    const arma::rowvec nLR { nx / nl,  ny / nl};
    const arma::rowvec nRL {-nx / nl, -ny / nl};

    // construct left state
    const double rhoL = uniform( 0.0, 1.0);
    const double uL = uniform(-1.0, 1.0);
    const double vL = uniform(-1.0, 1.0);
    const double EL = uniform( 0.0, 1.0) + 0.5 * (uL * uL + vL * vL);
    const arma::vec UL {rhoL, rhoL * uL, rhoL * vL, rhoL * EL};

    // construct right state
    const double rhoR = uniform( 0.0, 1.0);
    const double uR   = uniform(-1.0, 1.0);
    const double vR   = uniform(-1.0, 1.0);
    const double ER   = uniform( 0.0, 1.0) + 0.5 * (uR * uR + vR * vR);
    const arma::vec UR {rhoR, rhoR * uR, rhoR * vR, rhoR * ER};

    // temporary variable to store wave speed, not used here!
    double s;

    // compute Roe flux from left to right
    arma::vec FHatLR(4);
    problem.computeRoeFlux(UL, UR, nLR, FHatLR, s);

    // compute Roe flux from right to left
    arma::vec FHatRL(4);
    problem.computeRoeFlux(UR, UL, nRL, FHatRL, s);

    // return error
    return arma::norm(FHatLR + FHatRL);
}

double supersonicTestError(double gamma, const EulerDefaultBase &problem)
{
    // construct normal that points north-east
    const double nx = uniform(0.0, 1.0);
    const double ny = uniform(0.0, 1.0);
    const double nl = std::hypot(nx, ny);
    const arma::rowvec n {nx / nl, ny / nl};

    // construct supersonic state on the left
    const double rhoL = uniform(0.0, 1.0);
    const double uL   = uniform(0.8, 1.0);
    const double vL   = uniform(0.8, 1.0);
    const double eL   = uniform(0.0, 1.0);
    const double EL   = eL + 0.5 * (uL * uL + vL * vL);

    const arma::vec UL {rhoL, rhoL * uL, rhoL * vL, rhoL * EL};

    // construct supersonic state on the right
    const double rhoR = uniform(0.0, 1.0);
    const double uR   = uniform(0.8, 1.0);
    const double vR   = uniform(0.8, 1.0);
    const double eR   = uniform(0.0, 1.0);
    const double ER   = eR + 0.5 * (uR * uR + vR * vR);

    const arma::vec UR {rhoR, rhoR * uR, rhoR * vR, rhoR * ER};

    // compute Roe flux
    arma::vec FHat(4);
    double s;
    problem.computeRoeFlux(UL, UR, n, FHat, s);

    // compute analytical flux on the left
    const double pL = (gamma - 1.0) * rhoL * eL;
    const double HL = EL + pL / rhoL;

    const arma::vec FL {
            n[0] * (rhoL * uL)           + n[1] * (rhoL * vL),
            n[0] * (rhoL * uL * uL + pL) + n[1] * (rhoL * uL * vL),
            n[0] * (rhoL * uL * vL)      + n[1] * (rhoL * vL * vL + pL),
            n[0] * (rhoL * uL * HL)      + n[1] * (rhoL * vL * HL)};

    // return error
    return arma::norm(FHat - FL);
}

int main()
{
    // random seed
    std::srand(1279);

    // this choice is gamma is very important for supersonic test!
    const double gamma = 1.4;

    // set up problem
    EulerDefaultBase problem;
    problem.setSpecificHeatRatio(gamma);

    // report errors
    std::cout << "L2 errors in" << std::endl
              << "  Consistency test: " << std::scientific
              << consistencyTestError(gamma, problem) << std::endl
              << "  Flip test       : " << std::scientific
              << flipTestError(problem) << std::endl
              << "  Supersonic test : " << std::scientific
              << supersonicTestError(gamma, problem) << std::endl;

    return 0;
}
