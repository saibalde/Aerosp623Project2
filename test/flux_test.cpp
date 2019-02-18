#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "flux.hpp"

static const double gamma_ = 1.4;

double uniform(double a, double b)
{
    return a + ((b - a) * std::rand() / RAND_MAX);
}

void consistencyTest()
{
    const double rho = uniform( 0.0, 1.0);
    const double u   = uniform(-1.0, 1.0);
    const double v   = uniform(-1.0, 1.0);
    const double e   = uniform( 0.0, 1.0);

    const double nx  = uniform(-1.0, 1.0);
    const double ny  = uniform(-1.0, 1.0);
    const double nl  = std::hypot(nx, ny);

    const double E = e + 0.5 * (u * u + v * v);
    const double p = (gamma_ - 1.0) * rho * e;
    const double H = E + p / rho;

    const std::vector<double> U {rho, rho * u, rho * v, rho * E};
    const std::vector<double> n {nx / nl, ny / nl};
    const std::vector<double> F {
            n[0] * (rho * u)         + n[1] * (rho * v),
            n[0] * (rho * u * u + p) + n[1] * (rho * u * v),
            n[0] * (rho * u * v)     + n[1] * (rho * v * v + p),
            n[0] * (rho * u * H)     + n[1] * (rho * v * H)};

    std::vector<double> FHat(4);
    roeFlux(U, U, n, FHat);

    double error = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        double temp = F[i] - FHat[i];
        error += temp * temp;
    }
    error = std::sqrt(error);

    std::cout << "Consistency test: L2 error = " << error << std::endl;
}

void flipTest()
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

    const std::vector<double> UL {rhoL, rhoL * uL, rhoL * vL, rhoL * EL};
    const std::vector<double> UR {rhoR, rhoR * uR, rhoR * vR, rhoR * ER};

    const std::vector<double> nLR { nx / nl,  ny / nl};
    const std::vector<double> nRL {-nx / nl, -ny / nl};

    std::vector<double> FHatLR(4);
    roeFlux(UL, UR, nLR, FHatLR);

    std::vector<double> FHatRL(4);
    roeFlux(UR, UL, nRL, FHatRL);

    double error = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        double temp = FHatLR[i] + FHatRL[i];
        error += temp * temp;
    }
    error = std::sqrt(error);

    std::cout << "Anti-symmetry test: L2 error = " << error << std::endl;
}

void supersonicTest()
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

    const std::vector<double> UL {rhoL, rhoL * uL, rhoL * vL, rhoL * EL};
    const std::vector<double> UR {rhoR, rhoR * uR, rhoR * vR, rhoR * ER};

    const std::vector<double> n {1.0, 0.0};

    std::vector<double> FHat(4);
    roeFlux(UL, UR, n, FHat);

    const double pL = (gamma_ - 1.0) * rhoL * eL;
    const double HL = EL + pL / rhoL;

    const std::vector<double> FL {
            n[0] * (rhoL * uL)           + n[1] * (rhoL * vL),
            n[0] * (rhoL * uL * uL + pL) + n[1] * (rhoL * uL * vL),
            n[0] * (rhoL * uL * vL)      + n[1] * (rhoL * vL * vL + pL),
            n[0] * (rhoL * uL * HL)      + n[1] * (rhoL * vL * HL)};

    double error = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        double temp = FHat[i] - FL[i];
        error += temp * temp;
    }
    error = std::sqrt(error);

    std::cout << "Supersonic test: L2 error = " << error << std::endl;
}

int main()
{
    std::srand(0);

    consistencyTest();
    flipTest();
    supersonicTest();

    return 0;
}
