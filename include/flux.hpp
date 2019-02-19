#ifndef FLUX_HPP
#define FLUX_HPP

#include <armadillo>

void roeFlux(const arma::vec &uL, const arma::vec &uR,
             const arma::vec &n,  arma::vec &F);

#endif // FLUX_HPP
