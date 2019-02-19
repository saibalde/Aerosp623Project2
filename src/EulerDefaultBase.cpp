#include "Conv2D/EulerDefaultBase.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>

#include <cmath>

void EulerDefaultBase::computeFreeStreamState()
{
    const double cv = R_ / (gamma_ - 1.0);
    const double e = cv * Tt_;
    const double rho = pt_ / ((gamma_ - 1.0) * e);
    const double c = std::sqrt(gamma_ * (gamma_ - 1.0) * e);
    const double q = MInf_ * c;
    const double E = e + 0.5 * rho * q * q;
    const double u = q;
    const double v = 0.0;

    Ufree_.set_size(4);
    Ufree_(0) = rho;
    Ufree_(1) = rho * u;
    Ufree_(2) = rho * v;
    Ufree_(3) = rho * E;
}

void EulerDefaultBase::initialize()
{
    const arma::uword numElem = mesh_.nElemTot;

    Tt_ = 1.0 + 0.5 * (gamma_ - 1.0) * MInf_ * MInf_;
    pt_ = std::pow(Tt_, gamma_ / (gamma_ - 1.0));

    computeFreeStreamState();

    U_.set_size(4, numElem);
    for (arma::uword i = 0; i < numElem; ++i)
    {
        U_.col(i) = Ufree_;
    }
}

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

void EulerDefaultBase::computeRoeFlux(const arma::vec &UL, const arma::vec &UR,
                                      const arma::rowvec &n, arma::vec &F,
                                      double &s) const
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

    s = std::max(std::max(absLambda1, absLambda2), absLambda3);
}

void EulerDefaultBase::applyFreeStreamBC(const arma::vec &UInt,
                                         const arma::rowvec &n,
                                         arma::vec &F, double &s) const
{
    computeRoeFlux(UInt, Ufree_, n, F, s);
}

void EulerDefaultBase::applyInvisidWallBC(const arma::vec &UInt,
                                          const arma::rowvec &n,
                                          arma::vec &F, double &s) const
{
    const double rho = UInt(0);
    const double u   = UInt(1) / rho;
    const double v   = UInt(2) / rho;
    const double E   = UInt(3) / rho;

    const double p = u * n(0) + v * n(1);

    const double ub = u - p * n(0);
    const double vb = v - p * n(0);

    const double pb = (gamma_ - 1.0) * rho * (E - 0.5 * (ub * ub + vb * vb));

    F(0) = 0.0;
    F(1) = pb * n(0);
    F(2) = pb * n(1);
    F(3) = 0.0;

    s = 0.0;
}

void EulerDefaultBase::applyInflowBC(const arma::vec &UInt,
                                     const arma::rowvec &n,
                                     arma::vec &F,
                                     double &s) const
{
    const double rho = UInt(0);
    const double u   = UInt(1) / rho;
    const double v   = UInt(2) / rho;
    const double E   = UInt(3) / rho;

    const double p = (gamma_ - 1.0) * rho * (E - 0.5 * (u * u + v * v));
    const double c = std::sqrt(gamma_ * p / rho);

    const double w = u * n(0) + v * n(1);
    const double J = w + 2.0 * c / (gamma_ - 1);

    const double d = n(0);

    const double A = gamma_ * R_ * Tt_ * d * d - 0.5 * (gamma_ - 1.0) * J * J;
    const double B = 4.0 * gamma_ * R_ * Tt_ * d / (gamma_ - 1.0);
    const double C = 4.0 * gamma_ * R_ * Tt_ / ((gamma_ - 1.0) * (gamma_ - 1.0))
                   - J * J;

    const double Delta = B * B - 4.0 * A * C;
    if (Delta < 0)
    {
        throw std::runtime_error("Error in computing inflow flux");
    }

    const double Mb1 = (-B - std::sqrt(Delta)) / (2.0 * A);
    const double Mb2 = (-B + std::sqrt(Delta)) / (2.0 * A);

    const double MbMin = std::min(Mb1, Mb2);
    const double MbMax = std::max(Mb1, Mb2);

    const double Mb = MbMin < 0 ? MbMax : MbMax;

    if (Mb < 0)
    {
        throw std::runtime_error("Mach number cannot be negative!");
    }

    const double Tb   = Tt_ / (1.0 + 0.5 * (gamma_ - 1.0) * (Mb * Mb));
    const double pb   = pt_ * std::pow(Tb / Tt_, gamma_ / (gamma_ - 1.0));
    const double rhob = pb / (R_ * Tb);
    const double cb   = std::sqrt(gamma_ * pb / rhob);
    const double ub   = Mb * cb;
    const double vb   = 0.0;
    const double Eb   = pb / ((gamma_ - 1.0) * rhob) + 0.5 * (ub * ub + vb * vb);
    const double Hb   = Eb + pb / rhob;

    F(0) = n(0) * (rhob * ub)           + n(1) * (rhob * vb);
    F(1) = n(0) * (rhob * ub * ub + pb) + n(1) * (rhob * ub * vb);
    F(2) = n(0) * (rhob * ub * vb)      + n(1) * (rhob * vb * vb + pb);
    F(3) = n(0) * (rhob * ub * Hb)      + n(1) * (rhob * vb * Hb);

    s = cb + std::abs(ub * n(0) + vb * n(1));
}

void EulerDefaultBase::applyOutflowBC(const arma::vec &UInt,
                                      const arma::rowvec &n,
                                      arma::vec &F, double &s) const
{
    const double rho = UInt(0);
    const double u   = UInt(1) / rho;
    const double v   = UInt(2) / rho;
    const double E   = UInt(3) / rho;

    const double p = (gamma_ - 1) * rho * (E - 0.5 * (u * u + v * v));
    const double c = std::sqrt(gamma_ * p / rho);
    const double w = u * n(0) + v * n(1);
    const double J = w + 2.0 * c / (gamma_ - 1.0);
    const double S = p / std::pow(rho, gamma_);

    const double rhob = std::pow(pInf_ / S, 1.0 / gamma_);
    const double cb   = std::sqrt(gamma_ * pInf_ / rhob);
    const double wb   = J - 2.0 * cb / (gamma_ - 1.0);
    const double ub   = u + (wb - w) * n(0);
    const double vb   = v + (wb - w) * n(1);
    const double Eb   = pInf_ / ((gamma_ - 1.0) * rhob) + 0.5 * (ub * ub + vb * vb);
    const double Hb   = Eb + pInf_ / rhob;

    F(0) = n(0) * (rhob * ub)              + n(1) * (rhob * vb);
    F(1) = n(0) * (rhob * ub * ub + pInf_) + n(1) * (rhob * ub * vb);
    F(2) = n(0) * (rhob * ub * vb)         + n(1) * (rhob * vb * vb + pInf_);
    F(3) = n(0) * (rhob * ub * Hb)         + n(1) * (rhob * vb * Hb);

    s = cb + std::abs(ub * n(0) + vb * n(1));
}

double EulerDefaultBase::computeFirstOrderResidual(const arma::mat &U,
                                                   arma::mat &R,
                                                   arma::vec &S) const
{
    R.zeros();
    S.zeros();

    arma::uword numIFace = mesh_.I2E.n_rows;

    for (arma::uword i = 0; i < numIFace; ++i)
    {
        const arma::uword elemL = mesh_.I2E(i, 0) - 1;
        const arma::uword elemR = mesh_.I2E(i, 2) - 1;

        const arma::rowvec n = mesh_.In.row(i);
        const double l = mesh_.Il(i);

        arma::vec F(4);
        double s;
        computeRoeFlux(U.col(elemL), U.col(elemR), n, F, s);

        R.col(elemL) += l * F;
        R.col(elemR) -= l * F;

        S(elemL) += l * s;
        S(elemR) += l * s;
    }

    arma::uword iBFace = 0;

    arma::uword nBFaceBottom = mesh_.nBFace(0);
    for (arma::uword i = 0; i < nBFaceBottom; ++i)
    {
        if (mesh_.B2E(iBFace, 2) != 1)
        {
            throw std::runtime_error("Something is wrong with the mesh");
        }

        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec F(4);
        double s;
        computeBottomFlux(U.col(elem), n, F, s);

        R.col(elem) += l * F;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceRight = mesh_.nBFace(1);
    for (arma::uword i = 0; i < nBFaceRight; ++i)
    {
        if (mesh_.B2E(iBFace, 2) != 2)
        {
            throw std::runtime_error("Something is wrong with the mesh");
        }

        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec F(4);
        double s;
        computeRightFlux(U.col(elem), n, F, s);

        R.col(elem) += l * F;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceTop = mesh_.nBFace(2);
    for (arma::uword i = 0; i < nBFaceTop; ++i)
    {
        if (mesh_.B2E(iBFace, 2) != 3)
        {
            throw std::runtime_error("Something is wrong with the mesh");
        }

        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec F(4);
        double s;
        computeTopFlux(U.col(elem), n, F, s);

        R.col(elem) += l * F;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceLeft = mesh_.nBFace(3);
    for (arma::uword i = 0; i < nBFaceLeft; ++i)
    {
        if (mesh_.B2E(iBFace, 2) != 4)
        {
            throw std::runtime_error("Something is wrong with the mesh");
        }

        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec F(4);
        double s;

        computeLeftFlux(U.col(elem), n, F, s);

        R.col(elem) += l * F;
        S(elem) += l * s;

        iBFace += 1;
    }

    if (iBFace != mesh_.B2E.n_rows)
    {
        throw std::runtime_error("Something is wrong with the mesh");
    }

    return arma::abs(R).max();
}

void EulerDefaultBase::firstOrderSolver(arma::uword numIter)
{
    arma::uword numElem = mesh_.nElemTot;

    arma::mat R(4, numElem);
    arma::vec S(numElem);

    // compute initial residual and output
    double residual = computeFirstOrderResidual(U_, R, S);
    std::cout << 0 << " "
              << std::scientific << std::setprecision(15) << residual
              << std::endl;

    for (arma::uword i = 0; i < numIter; ++i)
    {
        // esitmate timesteps
        arma::vec dtOverA = 2.0 * CFL_ / S;

        // RK2 step 1
        arma::mat Utemp(4, numElem);
        for (arma::uword i = 0; i < numElem; ++i)
        {
            Utemp.col(i) = U_.col(i) - dtOverA(i) * R.col(i);
        }

        // RK2 step 2
        computeFirstOrderResidual(Utemp, R, S);
        for (arma::uword i = 0; i < numElem; ++i)
        {
            U_.col(i) = 0.5 * (U_.col(i) + Utemp.col(i) - dtOverA(i) * R.col(i));
        }

        // prepare residuals for next timestep
        residual = computeFirstOrderResidual(U_, R, S);

        // output residual
        std::cout << i + 1 << " "
                  << std::scientific << std::setprecision(15) << residual
                  << std::endl;
    }
}

void EulerDefaultBase::firstOrderSolver(double tolerance)
{
    arma::uword numElem = mesh_.nElemTot;

    arma::mat R(4, numElem);
    arma::vec S(numElem);

    // compute initial residual and output
    arma::uword numIter = 0;
    double residual = computeFirstOrderResidual(U_, R, S);
    std::cout << numIter << " "
              << std::scientific << std::setprecision(15) << residual
              << std::endl;

    while (residual > tolerance)
    {
        // esitmate timesteps
        arma::vec dtOverA = 2.0 * CFL_ / S;

        // RK2 step 1
        arma::mat Utemp(4, numElem);
        for (arma::uword i = 0; i < numElem; ++i)
        {
            Utemp.col(i) = U_.col(i) - dtOverA(i) * R.col(i);
        }

        // RK2 step 2
        computeFirstOrderResidual(Utemp, R, S);
        for (arma::uword i = 0; i < numElem; ++i)
        {
            U_.col(i) = 0.5 * (U_.col(i) + Utemp.col(i) - dtOverA(i) * R.col(i));
        }

        // prepare residual for next timestep
        residual = computeFirstOrderResidual(U_, R, S);

        ++numIter;

        // output residuals
        std::cout << numIter << " "
                  << std::scientific << std::setprecision(15) << residual
                  << std::endl;
    }
}

void EulerDefaultBase::output() const
{
    const std::string fileName = "state_" + std::to_string(stepNum_) + ".dat";
    U_.save(fileName, arma::raw_ascii);
}
