#include <iostream>

#include <cstdlib>

#include <armadillo>

#include "mesh2d.hpp"
#include "euler2d.hpp"

double uniform(double a, double b)
{
    return a + ((b - a) * std::rand() / RAND_MAX);
}

void consistencyTest(double gamma, const Euler2d &problem)
{
    const double rho = uniform( 0.0, 1.0);
    const double u   = uniform(-1.0, 1.0);
    const double v   = uniform(-1.0, 1.0);
    const double e   = uniform( 0.0, 1.0);

    const double nx  = uniform(-1.0, 1.0);
    const double ny  = uniform(-1.0, 1.0);
    const double nl  = std::hypot(nx, ny);

    const double E = e + 0.5 * (u * u + v * v);
    const double p = (gamma - 1.0) * rho * e;
    const double H = E + p / rho;

    const arma::vec U {rho, rho * u, rho * v, rho * E};
    const arma::rowvec n {nx / nl, ny / nl};
    const arma::vec F {
            n[0] * (rho * u)         + n[1] * (rho * v),
            n[0] * (rho * u * u + p) + n[1] * (rho * u * v),
            n[0] * (rho * u * v)     + n[1] * (rho * v * v + p),
            n[0] * (rho * u * H)     + n[1] * (rho * v * H)};

    arma::vec FHat(4);
    double s;
    problem.computeRoeFlux(U, U, n, FHat, s);

    const double error = arma::norm(F - FHat);

    std::cout << "Consistency test: L2 error = " << error << std::endl;
}

void flipTest(const Euler2d &problem)
{
    const double rhoL = uniform( 0.0, 1.0);
    const double uL   = uniform(-1.0, 1.0);
    const double vL   = uniform(-1.0, 1.0);
    const double EL   = uniform( 0.0, 1.0) + 0.5 * (uL * uL + vL * vL);

    const double rhoR = uniform( 0.0, 1.0);
    const double uR   = uniform(-1.0, 1.0);
    const double vR   = uniform(-1.0, 1.0);
    const double ER   = uniform( 0.0, 1.0) + 0.5 * (uR * uR + vR * vR);

    const double nx  = uniform(-1.0, 1.0);
    const double ny  = uniform(-1.0, 1.0);
    const double nl  = std::hypot(nx, ny);

    const arma::vec UL {rhoL, rhoL * uL, rhoL * vL, rhoL * EL};
    const arma::vec UR {rhoR, rhoR * uR, rhoR * vR, rhoR * ER};

    const arma::rowvec nLR { nx / nl,  ny / nl};
    const arma::rowvec nRL {-nx / nl, -ny / nl};

    arma::vec FHatLR(4);
    arma::vec FHatRL(4);
    double s;

    problem.computeRoeFlux(UL, UR, nLR, FHatLR, s);
    problem.computeRoeFlux(UR, UL, nRL, FHatRL, s);

    const double error = arma::norm(FHatLR + FHatRL);

    std::cout << "Flip test: L2 error = " << error << std::endl;
}

void supersonicTest(double gamma, const Euler2d &problem)
{
    const double rhoL = uniform( 0.0, 1.0);
    const double uL   = 0.8;
    const double vL   = 0.0;
    const double eL   = uniform( 0.0, 1.0);

    const double rhoR = uniform( 0.0, 1.0);
    const double uR   = uniform(-1.0, 0.0);
    const double vR   = uniform(-1.0, 1.0);
    const double eR   = uniform( 0.0, 1.0);

    const double EL = eL + 0.5 * (uL * uL + vL * vL);
    const double ER = eR + 0.5 * (uR * uR + vR * vR);

    const arma::vec UL {rhoL, rhoL * uL, rhoL * vL, rhoL * EL};
    const arma::vec UR {rhoR, rhoR * uR, rhoR * vR, rhoR * ER};

    const arma::rowvec n {1.0, 0.0};

    arma::vec FHat(4);
    double s;
    problem.computeRoeFlux(UL, UR, n, FHat, s);

    const double pL = (gamma - 1.0) * rhoL * eL;
    const double HL = EL + pL / rhoL;

    const arma::vec FL {
            n[0] * (rhoL * uL)           + n[1] * (rhoL * vL),
            n[0] * (rhoL * uL * uL + pL) + n[1] * (rhoL * uL * vL),
            n[0] * (rhoL * uL * vL)      + n[1] * (rhoL * vL * vL + pL),
            n[0] * (rhoL * uL * HL)      + n[1] * (rhoL * vL * HL)};

    const double error = arma::norm(FHat - FL);

    std::cout << "Supersonic test: L2 error = " << error << std::endl;
}

int main()
{
    std::srand(0);

    const double gamma = 1.4;
    const double R = 1.0;
    const double MInf = 0.5;
    const double pInf = 1.0;

    const Mesh2d mesh("test.gri");
    const Euler2d problem(mesh, gamma, R, MInf, pInf);

    consistencyTest(gamma, problem);
    flipTest(problem);
    supersonicTest(gamma, problem);

    return 0;
}
