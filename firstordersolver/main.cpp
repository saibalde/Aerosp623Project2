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

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        throw std::runtime_error("Name of mesh file not supplied");
    }

    Mesh mesh;
    mesh.readFromFile(argv[1]);
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

    const double tolerance = 1.0e-07;
    problem.firstOrderSolver(tolerance);

    return 0;
}
