#ifndef EULERDEFUALTBASE_HPP
#define EULERDEFAULTBASE_HPP

#include <armadillo>

#include "Conv2D/Mesh.hpp"

class EulerDefaultBase
{
public:
    EulerDefaultBase() : stepNum_(0)
    {
    }

    virtual ~EulerDefaultBase() = default;

    void setMesh(const Mesh &mesh)
    {
        mesh_ = mesh;
    }

    void setParams(double gamma, double R, double MInf, double pInf, double CFL)
    {
        gamma_ = gamma;
        R_ = R;
        MInf_ = MInf;
        pInf_ = pInf;
        CFL_ = CFL;
    }

    void initialize();

    void computeRoeFlux(const arma::vec &UL, const arma::vec &UR,
                        const arma::rowvec &n, arma::vec &F, double &s) const;

    virtual void computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                                   arma::vec &F, double &s) const = 0;

    virtual void computeRightFlux(const arma::vec &UInt,  const arma::rowvec &n,
                                  arma::vec &F, double &s) const = 0;

    virtual void computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                                arma::vec &F, double &s) const = 0;

    virtual void computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                                 arma::vec &F, double &s) const = 0;

    double timestep();

    void output() const;

protected:
    void applyFreeStreamBC(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &F, double &s) const;

    void applyInvisidWallBC(const arma::vec &UInt, const arma::rowvec &n,
                            arma::vec &F, double &s) const;

    void applyInflowBC(const arma::vec &UInt, const arma::rowvec &n,
                       arma::vec &F, double &s) const;

    void applyOutflowBC(const arma::vec &UInt, const arma::rowvec &n,
                        arma::vec &F, double &s) const;

private:
    arma::uword stepNum_;

    Mesh mesh_;

    double gamma_;
    double R_;
    double MInf_;
    double pInf_;
    double CFL_;

    double pt_;
    double Tt_;

    arma::vec Ufree_;
    arma::mat U_;

    void computeFreeStreamState();

    double computeResidual(const arma::mat &U, arma::mat &R,
                           arma::vec &S) const;
};

#endif // EULERDEFAULTBASE_HPP
