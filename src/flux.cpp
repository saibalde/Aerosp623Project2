#include "flux.hpp"

#include <cmath>

static const double gamma_ = 1.4;

double entropyFix(double epsilon, double lambda)
{
    if (std::abs(lambda) < epsilon)
    {
        return 0.5 * (epsilon * epsilon + lambda * lambda) / epsilon;
    }
    else
    {
        return std::abs(lambda);
    }
}

void roeFlux(const arma::vec &UL, const arma::vec &UR,
             const arma::vec &n,  arma::vec &F)
{
    // left state
    const double rhoL = UL(0);
    const double uL   = UL(1) / rhoL;
    const double vL   = UL(2) / rhoL;
    const double EL   = UL(3) / rhoL;
    const double pL   = (gamma_ - 1.0) * rhoL * (EL - 0.5 * (uL * uL + vL * vL));
    const double HL   = EL + pL / rhoL;

    arma::vec FL(4);
    FL(0) = n(0) * (rhoL * uL)           + n(1) * (rhoL * vL);
    FL(1) = n(0) * (rhoL * uL * uL + pL) + n(1) * (rhoL * uL * vL);
    FL(2) = n(0) * (rhoL * uL * vL)      + n(1) * (rhoL * vL * vL + pL);
    FL(3) = n(0) * (rhoL * uL * HL)      + n(1) * (rhoL * vL * HL);

    // right state
    const double rhoR = UR(0);
    const double uR   = UR(1) / rhoR;
    const double vR   = UR(2) / rhoR;
    const double ER   = UR(3) / rhoR;
    const double pR   = (gamma_ - 1.0) * rhoR * (ER - 0.5 * (uR * uR + vR * vR));
    const double HR   = ER + (gamma_ - 1.0) * (ER - 0.5 * (uR * uR + vR * vR));

    arma::vec FR(4);
    FR(0) = n(0) * (rhoR * uR)           + n(1) * (rhoR * vR);
    FR(1) = n(0) * (rhoR * uR * uR + pR) + n(1) * (rhoR * uR * vR);
    FR(2) = n(0) * (rhoR * uR * vR)      + n(1) * (rhoR * vR * vR + pR);
    FR(3) = n(0) * (rhoR * uR * HR)      + n(1) * (rhoR * vR * HR);

    // roe averaged state
    const double sqrtRhoL = std::sqrt(rhoL);
    const double sqrtRhoR = std::sqrt(rhoR);

    const double u = (uL * sqrtRhoL + uR * sqrtRhoR) / (sqrtRhoL + sqrtRhoR);
    const double v = (vL * sqrtRhoL + vR * sqrtRhoR) / (sqrtRhoL + sqrtRhoR);
    const double H = (HL * sqrtRhoL + HR * sqrtRhoR) / (sqrtRhoL + sqrtRhoR);

    const double c = std::sqrt((gamma_ - 1.0) * (H - 0.5 * (u * u + v * v)));

    // wave speeds with entropy fix
    const double p = u * n(0) + v * n(1);
    const double epsilon = 0.05 * c;

    const double absLambda1 = entropyFix(epsilon, p + c);
    const double absLambda2 = entropyFix(epsilon, p - c);
    const double absLambda3 = entropyFix(epsilon, p);

    // average flux
    F = 0.5 * (FL + FR);

    // roe flux correction
    const double s1 = 0.5 * (absLambda1 + absLambda2);
    const double s2 = 0.5 * (absLambda1 - absLambda2);

    const double G1 = (gamma_ - 1.0) * (0.5 * (u * u + v * v) * (rhoR - rhoL) -
                                        u * (rhoR * uR - rhoL * uL) -
                                        v * (rhoR * vR - rhoL * vL) +
                                        (rhoR * ER - rhoL * EL));
    const double G2 = -p * (rhoR - rhoL) + (rhoR * uR - rhoL * uL) * n(0)
                                         + (rhoR * vR - rhoL * vL) * n(1);

    const double C1 = G1 / (c * c) * (s1 - absLambda3) + G2 / c * s2;
    const double C2 = G1 / c * s2 + (s1 - absLambda3) * G2;

    F(0) -= 0.5 * (absLambda3 * (rhoR - rhoL) + C1);
    F(1) -= 0.5 * (absLambda3 * (rhoR * uR - rhoL * uL) + C1 * u + C2 * n(0));
    F(2) -= 0.5 * (absLambda3 * (rhoR * vR - rhoL * vL) + C1 * v + C2 * n(1));
    F(3) -= 0.5 * (absLambda3 * (rhoR * ER - rhoL * EL) + C1 * H + C2 * p);
}
