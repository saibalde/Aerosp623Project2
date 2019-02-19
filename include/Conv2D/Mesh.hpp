#ifndef MESH_HPP
#define MESH_HPP

#include <string>

#include <armadillo>

struct Mesh
{
    void readFromFile(const std::string &inFile);

    void writeToFile(const std::string &outFile) const;

    void computeMatrices();

    void writeMatricesToFile(const std::string &matFile) const;

    arma::uword nNode;
    arma::uword nElemTot;
    arma::uword dim;

    arma::mat nodeCoordinates;

    arma::uword nBGroup;

    arma::uvec nBFace;
    arma::uvec nf;
    arma::field<std::string> title;
    arma::field<arma::umat> B2N;

    arma::uword order;
    std::string basis;
    arma::umat E2N;

    arma::mat  I2M;
    arma::umat I2E;
    arma::mat  In;
    arma::vec  Il;

    arma::mat  B2M;
    arma::umat B2E;
    arma::mat  Bn;
    arma::vec  Bl;

    arma::mat E2M;
    arma::vec E2A;
};

#endif // MESH_HPP
