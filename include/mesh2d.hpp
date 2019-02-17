#ifndef MESH2D_HPP
#define MESH2D_HPP

#include <string>
#include <vector>

class Mesh2d
{
public:
    Mesh2d(const std::string &inFile);

    ~Mesh2d() = default;

    void setupMatrices();

    void output(const std::string &outFile) const;

    void outputMatrices(const std::string &matFile) const;

    int nNode;
    int nElemTot;
    int dim;

    std::vector<double> nodeCoordinates;

    int nBGroup;

    std::vector<int> nBFace;
    std::vector<int> nf;
    std::vector<std::string> title;
    std::vector<std::vector<int> > B2N;

    int order;
    std::string basis;
    std::vector<int> E2N;

    std::vector<int> I2E;
    std::vector<int> B2E;

    std::vector<double> In;
    std::vector<double> Bn;

    std::vector<double> area;
};

#endif // MESH2D_HPP
