#ifndef MESH2D_HPP
#define MESH2D_HPP

#include <string>
#include <vector>

class Mesh2d
{
public:
    Mesh2d(const std::string &gridFile);

    ~Mesh2d() = default;

    void output(const std::string &gridFile) const;

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
};

#endif // MESH2D_HPP
