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

void roeCorrection(double u, double v, double H,
                   const arma::vec &deltaU, arma::vec &deltaF)
{
    const double qSquare = u * u + v * v;
    const double cSquare = (gamma_ - 1.0) * (H - 0.5 * (u * u + v * v));

    arma::mat A(4, 4, arma::fill::none);
    A(0, 0) = 0.0;
    A(1, 0) = -u * u + (gamma_ - 1.0) / 2.0 * qSquare;
    A(2, 0) = -u * v;
    A(3, 0) = (gamma_ - 2.0) / 2.0 * u * qSquare - u * cSquare / (gamma_ - 1.0);
    A(0, 1) = 1.0;
    A(1, 1) = (3.0 - gamma_) * u;
    A(2, 1) = v;
    A(3, 1) = H - (gamma_ - 1.0) * u * u;
    A(0, 2) = 0.0;
    A(1, 2) = -(gamma_ - 1.0) * v;
    A(2, 2) = u;
    A(3, 2) = -(gamma_ - 1.0) * u * v;
    A(0, 3) = 0.0;
    A(1, 3) = gamma_ - 1.0;
    A(2, 3) = 0.0;
    A(3, 3) = gamma_ * u;

    arma::cx_vec eigVal;
    arma::cx_mat eigVec;
    arma::eig_gen(eigVal, eigVec, A);

    if (arma::norm(arma::imag(eigVal)) > 1.0e-12 ||
        arma::norm(arma::vectorise(arma::imag(eigVec))) > 1.0e-12)
    {
        std::cout << "Warning: Encountered complex eigenvalue decomposition "
                  << "in computation of Roe flux" << std::endl;
    }

    arma::mat absA = arma::abs(eigVec) * arma::diagmat(arma::abs(eigVal)) *
                     arma::inv(arma::abs(eigVec));

    deltaF = absA * deltaU;
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

    const double u   = weightedAverage(sqrtRhoL, sqrtRhoR, uL, uR);
    const double v   = weightedAverage(sqrtRhoL, sqrtRhoR, vL, vR);
    const double H   = weightedAverage(sqrtRhoL, sqrtRhoR, HL, HR);

    arma::vec deltaU(4, arma::fill::none);
    arma::vec deltaF(4, arma::fill::none);

    for (int i = 0; i < 4; ++i)
    {
        deltaU(i) = UR[i] - UL[i];
    }

    roeCorrection(u, v, H, deltaU, deltaF);

    for (int i = 0; i < 4; ++i)
    {
        F[i] = 0.5 * (FL[i] + FR[i] - deltaF(i));
    }
}
