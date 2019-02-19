#ifndef EULER2D_HPP
#define EULER2D_HPP

#include <armadillo>

#include "mesh2d.hpp"

class Euler2d
{
public:
    Euler2d(Mesh2d mesh, double gamma, double R, double MInf, double pInf)
        : mesh_(mesh),
          gamma_(gamma),
          R_(R),
          MInf_(MInf),
          pInf_(pInf),
          stepNum_(0)
    {
    }

    void initialize();

    void setBoundaryStates();

    void timestep();

    void output() const;

    void computeRoeFlux(const arma::vec &UL, const arma::vec &UR,
                        const arma::rowvec &n, arma::vec &F, double &s) const;

    void computeResidual(const arma::mat &U, arma::mat &R, arma::vec &S) const;

    arma::mat U_;

private:
    Mesh2d mesh_;
    double gamma_;
    double R_;
    double MInf_;
    double pInf_;

    arma::uword stepNum_;

    double pt_;
    double Tt_;

    arma::mat Ubottom_;
    arma::mat Uright_;
    arma::mat Utop_;
    arma::mat Uleft_;

    void computeFreeStreamState(arma::vec &U) const;
};

#endif // EULER2D_HPP
