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


void Euler::computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                              arma::vec &F, double &s) const
{
    applyFreeStreamBC(UInt, n, F, s);
}

void Euler::computeRightFlux(const arma::vec &UInt, const arma::rowvec &n,
                             arma::vec &F, double &s) const
{
    applyFreeStreamBC(UInt, n, F, s);
}

void Euler::computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &F, double &s) const
{
    applyFreeStreamBC(UInt, n, F, s);
}

void Euler::computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                            arma::vec &F, double &s) const
{
    applyFreeStreamBC(UInt, n, F, s);
}

int main()
{
    Mesh mesh;
    mesh.readFromFile("bump0.gri");
    mesh.computeMatrices();

    const double gamma = 1.4;
    const double R = 1.0;
    const double MInf = 0.5;
    const double pInf = 1.0;
    const double CFL = 0.5;

    Euler problem;
    problem.setMesh(mesh);
    problem.setParams(gamma, R, MInf, pInf, CFL);
    problem.initialize();

    const arma::uword numIter = 1000;
    problem.firstOrderSolver(numIter);

    return 0;
}
