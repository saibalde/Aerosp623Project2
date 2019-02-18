#include "flux.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>

#include <armadillo>

static const double gamma_ = 1.4;

double weightedAverage(double wL, double wR, double qL, double qR)
{
    return (wL * qL + wR * qR) / (wL + wR);
}

void roeFlux(const std::vector<double> &UL, const std::vector<double> &UR,
             const std::vector<double> &n,  std::vector<double> &F)
{
    // size check

    if (UL.size() != 4 || UR.size() != 4 || n.size() != 2 || F.size() != 4)
    {
        throw std::logic_error("Dimension error in computing Roe flux");
    }

    // left state

    const double rhoL     = UL[0];
    const double uL       = UL[1] / rhoL;
    const double vL       = UL[2] / rhoL;
    const double EL       = UL[3] / rhoL;

    const double pL = (gamma_ - 1.0) * rhoL * (EL - 0.5 * (uL * uL + vL * vL));
    const double HL = EL + pL / rhoL;

    const std::vector<double> FL {
            n[0] * (rhoL * uL)           + n[1] * (rhoL * vL),
            n[0] * (rhoL * uL * uL + pL) + n[1] * (rhoL * uL * vL),
            n[0] * (rhoL * uL * vL)      + n[1] * (rhoL * vL * vL + pL),
            n[0] * (rhoL * uL * HL)      + n[1] * (rhoL * vL * HL)};

    const double sqrtRhoL = std::sqrt(rhoL);

    // right state

    const double rhoR     = UR[0];
    const double uR       = UR[1] / rhoR;
    const double vR       = UR[2] / rhoR;
    const double ER       = UR[3] / rhoR;

    const double pR = (gamma_ - 1.0) * rhoR * (ER - 0.5 * (uR * uR + vR * vR));
    const double HR = ER + pR / rhoR;

    const std::vector<double> FR {
            n[0] * (rhoR * uR)           + n[1] * (rhoR * vR),
            n[0] * (rhoR * uR * uR + pR) + n[1] * (rhoR * uR * vR),
            n[0] * (rhoR * uR * vR)      + n[1] * (rhoR * vR * vR + pR),
            n[0] * (rhoR * uR * HR)      + n[1] * (rhoR * vR * HR)};

    const double sqrtRhoR = std::sqrt(rhoR);

    // roe averaged state

    const double u = weightedAverage(sqrtRhoL, sqrtRhoR, uL, uR);
    const double v = weightedAverage(sqrtRhoL, sqrtRhoR, vL, vR);
    const double H = weightedAverage(sqrtRhoL, sqrtRhoR, HL, HR);
    const double c = std::sqrt((gamma_ - 1.0) * (H - 0.5 * (u * u + v * v)));

    arma::vec::fixed<4>    deltaU;
    arma::mat::fixed<4, 4> R;
    arma::vec::fixed<4>    absLambda;
    arma::vec::fixed<4>    deltaF;

    deltaU(0) = UR[0] - UL[0];
    deltaU(1) = UR[1] - UL[1];
    deltaU(2) = UR[2] - UL[2];
    deltaU(3) = UR[3] - UL[3];

    R(0, 0) = 1.0;
    R(1, 0) = u + c;
    R(2, 0) = v;
    R(3, 0) = H + u * c;
    R(0, 1) = 1.0;
    R(1, 1) = u - c;
    R(2, 1) = v;
    R(3, 1) = H - u * c;
    R(0, 2) = 0.0;
    R(1, 2) = 0.0;
    R(2, 2) = v;
    R(3, 2) = v * v;
    R(0, 3) = 1.0;
    R(1, 3) = u;
    R(2, 3) = v;
    R(3, 3) = 0.5 * (u * u + v * v);

    absLambda(0) = std::abs(u + c);
    absLambda(1) = std::abs(u - c);
    absLambda(2) = std::abs(u);
    absLambda(3) = std::abs(u);

    deltaF = -0.5 * (R * (arma::diagmat(absLambda) * (arma::inv(R) * deltaU)));

    for (int i = 0; i < 4; ++i)
    {
        F[i] = 0.5 * (FL[i] + FR[i]) + deltaF(i);
    }
}
