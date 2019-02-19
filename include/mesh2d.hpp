#ifndef MESH2D_HPP
#define MESH2D_HPP

#include <string>

#include <armadillo>

class Mesh2d
{
public:
    Mesh2d(const std::string &inFile);

    ~Mesh2d() = default;

    void setupMatrices();

    void output(const std::string &outFile) const;

    void outputMatrices(const std::string &matFile) const;

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

    arma::umat I2E;
    arma::umat B2E;

    arma::mat In;
    arma::mat Bn;

    arma::vec Il;
    arma::vec Bl;

    arma::vec area;
};

#endif // MESH2D_HPP
