#ifndef FLUX_HPP
#define FLUX_HPP

#include <vector>

void roeFlux(const std::vector<double> &uL, const std::vector<double> &uR,
             const std::vector<double> &n,  std::vector<double> &F);

#endif // FLUX_HPP
