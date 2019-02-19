#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <armadillo>

#include "Conv2D/Mesh.hpp"
#include "Conv2D/EulerDefaultBase.hpp"

class Euler : public EulerDefaultBase
{
public:
    Euler() : EulerDefaultBase()
    {
    }

    ~Euler() = default;

    void computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &F, double &s) const;

    void computeRightFlux(const arma::vec &UInt, const arma::rowvec &n,
                          arma::vec &F, double &s) const;

    void computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                        arma::vec &F, double &s) const;

    void computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                         arma::vec &F, double &s) const;
};

void Euler::computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &F, double &s) const
{
    applyInvisidWallBC(UInt, n, F, s);
}

void Euler::computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                              arma::vec &F, double &s) const
{
    applyInvisidWallBC(UInt, n, F, s);
}

void Euler::computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                            arma::vec &F, double &s) const
{
    applyInflowBC(UInt, n, F, s);
}

void Euler::computeRightFlux(const arma::vec &UInt, const arma::rowvec &n,
                             arma::vec &F, double &s) const
{
    applyOutflowBC(UInt, n, F, s);
}

int main()
{
    Mesh mesh;
    mesh.readFromFile("bump0.gri");
    mesh.computeMatrices();

    Euler problem;
    problem.setMesh(mesh);
    problem.setParams(1.4, 1.0, 0.5, 1.0, 0.5);
    problem.initialize();
    problem.output();

    std::cout << "Residuals:" << std::endl;

    double residual = 2.0e-7;
    arma::uword iterCount = 0;

    while (residual >= 1.0e-7)
    {
        residual = problem.timestep();

        std::cout << iterCount << " "
                  << std::scientific << std::setprecision(15) << residual
                  << std::endl;

        ++iterCount;

    }

    return 0;
}
