#include "Conv2D/EulerDefaultBase.hpp"

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

void EulerDefaultBase::initialize(const std::string fileName)
{
    const arma::uword numElem = mesh_.nElemTot;

    Tt_ = 1.0 + 0.5 * (gamma_ - 1.0) * MInf_ * MInf_;
    pt_ = std::pow(Tt_, gamma_ / (gamma_ - 1.0));

    computeFreeStreamState();

    U_.load(fileName, arma::raw_ascii);

    if (U_.n_rows != 4 || U_.n_cols != numElem)
    {
        throw std::runtime_error("Could not set initial state from file");
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
                                         arma::vec &U, arma::vec &F,
                                         double &s) const
{
    U = Ufree_;
    computeRoeFlux(UInt, Ufree_, n, F, s);
}

void EulerDefaultBase::applyInvisidWallBC(const arma::vec &UInt,
                                          const arma::rowvec &n,
                                          arma::vec &U, arma::vec &F,
                                          double &s) const
{
    const double rho = UInt(0);
    const double u   = UInt(1) / rho;
    const double v   = UInt(2) / rho;
    const double E   = UInt(3) / rho;

    const double p = u * n(0) + v * n(1);

    const double ub = u - p * n(0);
    const double vb = v - p * n(1);

    const double pb = (gamma_ - 1.0) * rho * (E - 0.5 * (ub * ub + vb * vb));

    U(0) = rho;
    U(1) = rho * ub;
    U(2) = rho * vb;
    U(3) = rho * E;

    F(0) = 0.0;
    F(1) = pb * n(0);
    F(2) = pb * n(1);
    F(3) = 0.0;

    s = 0.0;
}

void EulerDefaultBase::applyInflowBC(const arma::vec &UInt,
                                     const arma::rowvec &n,
                                     arma::vec &U, arma::vec &F,
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

    U(0) = rhob;
    U(1) = rhob * ub;
    U(2) = rhob * vb;
    U(3) = rhob * Eb;

    F(0) = n(0) * (rhob * ub)           + n(1) * (rhob * vb);
    F(1) = n(0) * (rhob * ub * ub + pb) + n(1) * (rhob * ub * vb);
    F(2) = n(0) * (rhob * ub * vb)      + n(1) * (rhob * vb * vb + pb);
    F(3) = n(0) * (rhob * ub * Hb)      + n(1) * (rhob * vb * Hb);

    s = cb + std::abs(ub * n(0) + vb * n(1));
}

void EulerDefaultBase::applyOutflowBC(const arma::vec &UInt,
                                      const arma::rowvec &n,
                                      arma::vec &U, arma::vec &F,
                                      double &s) const
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

    U(0) = rhob;
    U(1) = rhob * ub;
    U(2) = rhob * vb;
    U(3) = rhob * Eb;

    F(0) = n(0) * (rhob * ub)              + n(1) * (rhob * vb);
    F(1) = n(0) * (rhob * ub * ub + pInf_) + n(1) * (rhob * ub * vb);
    F(2) = n(0) * (rhob * ub * vb)         + n(1) * (rhob * vb * vb + pInf_);
    F(3) = n(0) * (rhob * ub * Hb)         + n(1) * (rhob * vb * Hb);

    s = cb + std::abs(ub * n(0) + vb * n(1));
}

void EulerDefaultBase::computeBottomFlux(const arma::vec &UInt,
                                         const arma::rowvec &n,
                                         arma::vec &U, arma::vec &F,
                                         double &s) const
{
    applyFreeStreamBC(UInt, n, U, F, s);
}

void EulerDefaultBase::computeRightFlux(const arma::vec &UInt,
                                        const arma::rowvec &n,
                                        arma::vec &U, arma::vec &F,
                                        double &s) const
{
    applyFreeStreamBC(UInt, n, U, F, s);
}

void EulerDefaultBase::computeTopFlux(const arma::vec &UInt,
                                      const arma::rowvec &n,
                                      arma::vec &U, arma::vec &F,
                                      double &s) const
{
    applyFreeStreamBC(UInt, n, U, F, s);
}

void EulerDefaultBase::computeLeftFlux(const arma::vec &UInt,
                                       const arma::rowvec &n,
                                       arma::vec &U, arma::vec &F,
                                       double &s) const
{
    applyFreeStreamBC(UInt, n, U, F, s);
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
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeBottomFlux(U.col(elem), n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceRight = mesh_.nBFace(1);
    for (arma::uword i = 0; i < nBFaceRight; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeRightFlux(U.col(elem), n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceTop = mesh_.nBFace(2);
    for (arma::uword i = 0; i < nBFaceTop; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeTopFlux(U.col(elem), n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceLeft = mesh_.nBFace(3);
    for (arma::uword i = 0; i < nBFaceLeft; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;

        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;

        computeLeftFlux(U.col(elem), n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    return arma::abs(R).max();
}

void EulerDefaultBase::runFirstOrderSolver(arma::uword numIter,
                                           const std::string &residualFile)
{
    std::ofstream file(residualFile);

    arma::uword numElem = mesh_.nElemTot;

    arma::mat R(4, numElem);
    arma::vec S(numElem);

    // compute initial residual and output
    double residual = computeFirstOrderResidual(U_, R, S);
    file << 0 << " "
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
        file << i + 1 << " "
             << std::scientific << std::setprecision(15) << residual
             << std::endl;
    }
}

void EulerDefaultBase::runFirstOrderSolver(double tolerance,
                                           const std::string &residualFile)
{
    std::ofstream file(residualFile);

    arma::uword numElem = mesh_.nElemTot;

    arma::mat R(4, numElem);
    arma::vec S(numElem);

    // compute initial residual and output
    arma::uword numIter = 0;
    double residual = computeFirstOrderResidual(U_, R, S);
    file << numIter << " "
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
        file << numIter << " "
             << std::scientific << std::setprecision(15) << residual
             << std::endl;
    }
}

void EulerDefaultBase::computeGradients(const arma::mat &U, arma::mat &G) const
{
    G.zeros();

    const arma::uword nIFace = mesh_.Il.n_rows;

    for (arma::uword i = 0; i < nIFace; ++i)
    {
        const arma::uword elemL = mesh_.I2E(i, 0) - 1;
        const arma::uword elemR = mesh_.I2E(i, 2) - 1;
        const arma::rowvec n = mesh_.In.row(i);
        const double l = mesh_.Il(i);

        const arma::vec u = 0.5 * (U.col(elemL) + U.col(elemR));

        arma::vec g(8);
        g(arma::span(0, 3)) = l * n(0) * u;
        g(arma::span(4, 7)) = l * n(1) * u;

        G.col(elemL) += g;
        G.col(elemR) -= g;
    }

    arma::uword iBFace = 0;

    const arma::uword nBFaceBottom = mesh_.nBFace(0);

    for (arma::uword i = 0; i < nBFaceBottom; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeBottomFlux(U.col(elem), n, u, f, s);

        arma::vec g(8);
        g(arma::span(0, 3)) = l * n(0) * u;
        g(arma::span(4, 7)) = l * n(1) * u;

        G.col(elem) += g;

        iBFace += 1;
    }

    const arma::uword nBFaceRight = mesh_.nBFace(1);

    for (arma::uword i = 0; i < nBFaceRight; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeRightFlux(U.col(elem), n, u, f, s);

        arma::vec g(8);
        g(arma::span(0, 3)) = l * n(0) * u;
        g(arma::span(4, 7)) = l * n(1) * u;

        G.col(elem) += g;

        iBFace += 1;
    }

    const arma::uword nBFaceTop = mesh_.nBFace(2);

    for (arma::uword i = 0; i < nBFaceTop; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeTopFlux(U.col(elem), n, u, f, s);

        arma::vec g(8);
        g(arma::span(0, 3)) = l * n(0) * u;
        g(arma::span(4, 7)) = l * n(1) * u;

        G.col(elem) += g;

        iBFace += 1;
    }

    const arma::uword nBFaceLeft = mesh_.nBFace(3);

    for (arma::uword i = 0; i < nBFaceLeft; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeLeftFlux(U.col(elem), n, u, f, s);

        arma::vec g(8);
        g(arma::span(0, 3)) = l * n(0) * u;
        g(arma::span(4, 7)) = l * n(1) * u;

        G.col(elem) += g;

        iBFace += 1;
    }

    const arma::uword nElem = mesh_.nElemTot;

    for (arma::uword i = 0; i < nElem; ++i)
    {
        G.col(i) /= mesh_.E2A(i);
    }
}

double EulerDefaultBase::computeSecondOrderResidual(const arma::mat &U,
                                                    const arma::mat &G,
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

        const double xL = mesh_.I2M(i, 0) - mesh_.E2M(elemL, 0);
        const double yL = mesh_.I2M(i, 1) - mesh_.E2M(elemL, 1);
        const arma::vec gL = G.col(elemL);
        const arma::vec uL = U.col(elemL) + xL * gL(arma::span(0, 3))
                                          + yL * gL(arma::span(4, 7));

        const double xR = mesh_.I2M(i, 0) - mesh_.E2M(elemR, 0);
        const double yR = mesh_.I2M(i, 1) - mesh_.E2M(elemR, 1);
        const arma::vec gR = G.col(elemR);
        const arma::vec uR = U.col(elemR) + xR * gR(arma::span(0, 3))
                                          + yR * gR(arma::span(4, 7));

        arma::vec F(4);
        double s;
        computeRoeFlux(uL, uR, n, F, s);

        R.col(elemL) += l * F;
        R.col(elemR) -= l * F;

        S(elemL) += l * s;
        S(elemR) += l * s;
    }

    arma::uword iBFace = 0;

    arma::uword nBFaceBottom = mesh_.nBFace(0);
    for (arma::uword i = 0; i < nBFaceBottom; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        const double x = mesh_.B2M(iBFace, 0) - mesh_.E2M(elem, 0);
        const double y = mesh_.B2M(iBFace, 1) - mesh_.E2M(elem, 1);
        const arma::vec g = G.col(elem);
        const arma::vec uInt = U.col(elem) + x * g(arma::span(0, 3))
                                           + y * g(arma::span(4, 7));

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeBottomFlux(uInt, n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceRight = mesh_.nBFace(1);
    for (arma::uword i = 0; i < nBFaceRight; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        const double x = mesh_.B2M(iBFace, 0) - mesh_.E2M(elem, 0);
        const double y = mesh_.B2M(iBFace, 1) - mesh_.E2M(elem, 1);
        const arma::vec g = G.col(elem);
        const arma::vec uInt = U.col(elem) + x * g(arma::span(0, 3))
                                           + y * g(arma::span(4, 7));

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeRightFlux(uInt, n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceTop = mesh_.nBFace(2);
    for (arma::uword i = 0; i < nBFaceTop; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        const double x = mesh_.B2M(iBFace, 0) - mesh_.E2M(elem, 0);
        const double y = mesh_.B2M(iBFace, 1) - mesh_.E2M(elem, 1);
        const arma::vec g = G.col(elem);
        const arma::vec uInt = U.col(elem) + x * g(arma::span(0, 3))
                                           + y * g(arma::span(4, 7));

        arma::vec u(4);
        arma::vec f(4);
        double s;
        computeTopFlux(uInt, n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    arma::uword nBFaceLeft = mesh_.nBFace(3);
    for (arma::uword i = 0; i < nBFaceLeft; ++i)
    {
        const arma::uword elem = mesh_.B2E(iBFace, 0) - 1;
        const arma::rowvec n = mesh_.Bn.row(iBFace);
        const double l = mesh_.Bl(iBFace);

        const double x = mesh_.B2M(iBFace, 0) - mesh_.E2M(elem, 0);
        const double y = mesh_.B2M(iBFace, 1) - mesh_.E2M(elem, 1);
        const arma::vec g = G.col(elem);
        const arma::vec uInt = U.col(elem) + x * g(arma::span(0, 3))
                                           + y * g(arma::span(4, 7));

        arma::vec u(4);
        arma::vec f(4);
        double s;

        computeLeftFlux(uInt, n, u, f, s);

        R.col(elem) += l * f;
        S(elem) += l * s;

        iBFace += 1;
    }

    return arma::abs(R).max();
}

void EulerDefaultBase::runSecondOrderSolver(arma::uword numIter,
                                            const std::string &residualFile)
{
    std::ofstream file(residualFile);

    arma::uword numElem = mesh_.nElemTot;

    arma::mat G(8, numElem);
    arma::mat R(4, numElem);
    arma::vec S(numElem);

    // compute initial residual and output
    computeGradients(U_, G);
    double residual = computeSecondOrderResidual(U_, G, R, S);
    file << 0 << " "
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
        computeGradients(Utemp, G);
        computeSecondOrderResidual(Utemp, G, R, S);
        for (arma::uword i = 0; i < numElem; ++i)
        {
            U_.col(i) = 0.5 * (U_.col(i) + Utemp.col(i) - dtOverA(i) * R.col(i));
        }

        // prepare residuals for next timestep
        computeGradients(U_, G);
        residual = computeSecondOrderResidual(U_, G, R, S);

        // output residual
        file << i + 1 << " "
             << std::scientific << std::setprecision(15) << residual
             << std::endl;
    }
}

void EulerDefaultBase::runSecondOrderSolver(double tolerance,
                                            const std::string &residualFile)
{
    std::ofstream file(residualFile);

    arma::uword numElem = mesh_.nElemTot;

    arma::mat G(8, numElem);
    arma::mat R(4, numElem);
    arma::vec S(numElem);

    // compute initial residual and output
    arma::uword numIter = 0;
    computeGradients(U_, G);
    double residual = computeSecondOrderResidual(U_, G, R, S);
    file << numIter << " "
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
        computeGradients(Utemp, G);
        computeSecondOrderResidual(Utemp, G, R, S);
        for (arma::uword i = 0; i < numElem; ++i)
        {
            U_.col(i) = 0.5 * (U_.col(i) + Utemp.col(i) - dtOverA(i) * R.col(i));
        }

        // prepare residual for next timestep
        computeGradients(U_, G);
        residual = computeSecondOrderResidual(U_, G, R, S);

        ++numIter;

        // output residuals
        file << numIter << " "
             << std::scientific << std::setprecision(15) << residual
             << std::endl;
    }
}

double EulerDefaultBase::liftCoefficient() const
{
    const double h = 0.0625;

    const arma::umat  &B2E         = mesh_.B2E;
    const arma::mat   &Bn          = mesh_.Bn;
    const arma::vec   &Bl          = mesh_.Bl;
    const arma::uword nBFaceBottom = mesh_.nBFace(0);

    double cl = 0.0;

    for (arma::uword i = 0; i < nBFaceBottom; ++i)
    {
        const arma::uword elem = B2E(i, 0) - 1;

        const double rho = U_(0, elem);
        const double u   = U_(1, elem) / rho;
        const double v   = U_(2, elem) / rho;
        const double E   = U_(3, elem) / rho;

        const double p  = (gamma_ - 1.0) * rho * (E - 0.5 * (u * u + v * v));
        const double ny = Bn(i, 1);
        const double l  = Bl(i);

        cl += (p - pInf_) * ny * l;
    }

    cl = cl / (0.5 * gamma_ * pInf_ * MInf_ * MInf_ * h);

    return cl;
}

double EulerDefaultBase::dragCoefficient() const
{
    const double h = 0.0625;

    const arma::umat  &B2E         = mesh_.B2E;
    const arma::mat   &Bn          = mesh_.Bn;
    const arma::vec   &Bl          = mesh_.Bl;
    const arma::uword nBFaceBottom = mesh_.nBFace(0);

    double cd = 0.0;

    for (arma::uword i = 0; i < nBFaceBottom; ++i)
    {
        const arma::uword elem = B2E(i, 0) - 1;

        const double rho = U_(0, elem);
        const double u   = U_(1, elem) / rho;
        const double v   = U_(2, elem) / rho;
        const double E   = U_(3, elem) / rho;

        const double p  = (gamma_ - 1.0) * rho * (E - 0.5 * (u * u + v * v));
        const double nx = Bn(i, 0);
        const double l  = Bl(i);

        cd += (p - pInf_) * nx * l;
    }

    cd = cd / (0.5 * gamma_ * pInf_ * MInf_ * MInf_ * h);

    return cd;
}

double EulerDefaultBase::entropyError() const
{
    const double rhot = pt_ / (R_ * Tt_);
    const double st = pt_ / std::pow(rhot, gamma_);

    const arma::vec   &E2A    = mesh_.E2A;
    const arma::uword numElem = mesh_.nElemTot;

    double Es   = 0.0;
    double area = 0.0;

    for (arma::uword i = 0; i < numElem; ++i)
    {
        const double rho = U_(0, i);
        const double u   = U_(1, i) / rho;
        const double v   = U_(2, i) / rho;
        const double E   = U_(3, i) / rho;

        const double p = (gamma_ - 1.0) * rho * (E - 0.5 * (u * u + v * v));
        const double s = p / std::pow(rho, gamma_);

        Es   += (s / st - 1.0) * (s / st - 1.0) * E2A(i);
        area += E2A(i);
    }

    Es = Es / area;
    Es = std::sqrt(Es);

    return Es;
}

void EulerDefaultBase::writeValidationValuesTofile(
        const std::string &fileName) const
{
    std::ofstream file(fileName);

    file << "Lift coefficient = "
         << std::scientific << std::setprecision(15)
         << liftCoefficient() << std::endl;
    file << "Drag coefficient = "
         << std::scientific << std::setprecision(15)
         << dragCoefficient() << std::endl;
    file << "Entropy error = "
         << std::scientific << std::setprecision(15)
         << entropyError()    << std::endl;
}

void EulerDefaultBase::writePressureCoefficientsToFile(
        const std::string &fileName) const
{
    const arma::umat  &B2E         = mesh_.B2E;
    const arma::uword nBFaceBottom = mesh_.nBFace(0);

    arma::mat cp(nBFaceBottom, 2);

    const double den = 0.5 * gamma_ * MInf_ * MInf_;

    for (arma::uword i = 0; i < nBFaceBottom; ++i)
    {
        const arma::uword elem = B2E(i, 0) - 1;

        const double rho = U_(0, elem);
        const double u   = U_(1, elem) / rho;
        const double v   = U_(2, elem) / rho;
        const double E   = U_(3, elem) / rho;

        const double p  = (gamma_ - 1.0) * rho * (E - 0.5 * (u * u + v * v));

        cp(i, 0) = mesh_.B2M(i, 0);
        cp(i, 1) = (p - pInf_) / den;
    }

    cp.save(fileName, arma::raw_ascii);
}

void EulerDefaultBase::writeMachNumbersToFile(const std::string &fileName) const
{
    const arma::uword numNode = mesh_.nNode;
    const arma::uword numElem = mesh_.nElemTot;

    arma::vec M(numElem);

    for (arma::uword i = 0; i < numElem; ++i)
    {
        const double rho = U_(0, i);
        const double u   = U_(1, i) / rho;
        const double v   = U_(2, i) / rho;
        const double E   = U_(3, i) / rho;

        const double q   = std::sqrt(u * u + v * v);
        const double p = (gamma_ - 1.0) * rho * (E - 0.5 * q * q);
        const double c = std::sqrt(gamma_ * p / rho);

        M(i) = q / c;
    }

    std::ofstream file(fileName);

    file << "# vtk DataFile Version3.0" << std::endl;
    file << "Mach numbers" << std::endl;
    file << "ASCII" << std::endl;
    file << std::endl;

    file << "DATASET UNSTRUCTURED_GRID" << std::endl;
    file << "POINTS " << numNode << " double" << std::endl;
    for (arma::uword i = 0; i < numNode; ++i)
    {
        file << std::scientific << std::setprecision(15)
             << mesh_.nodeCoordinates(i, 0) << " "
             << std::scientific << std::setprecision(15)
             << mesh_.nodeCoordinates(i, 1) << " "
             << std::scientific << std::setprecision(15)
             << 0.0 << std::endl;;
    }
    file << std::endl;

    file << "CELLS " << numElem << " " << 4 * numElem << std::endl;
    for (arma::uword i = 0; i < numElem; ++i)
    {
        file << 3 << " "
             << mesh_.E2N(i, 0) - 1 << " "
             << mesh_.E2N(i, 1) - 1 << " "
             << mesh_.E2N(i, 2) - 1 << std::endl;
    }
    file << std::endl;

    file << "CELL_TYPES " << numElem << std::endl;
    for (arma::uword i = 0; i < numElem; ++i)
    {
        file << 5 << std::endl;
    }
    file << std::endl;

    file << "CELL_DATA " << numElem << std::endl;
    file << "SCALARS mach_number double " << 1 << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (arma::uword i = 0; i < numElem; ++i)
    {
        file << M(i) << std::endl;
    }
}

void EulerDefaultBase::writeStateToFile(const std::string &fileName) const
{
    U_.save(fileName, arma::raw_ascii);
}
