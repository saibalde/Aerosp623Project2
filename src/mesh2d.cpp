#include "mesh2d.hpp"

#include <iomanip>
#include <fstream>
#include <stdexcept>

Mesh2d::Mesh2d(const std::string &gridFile)
{
    std::ifstream file(gridFile);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open grid file");
    }

    std::string line;

    file >> nNode >> nElemTot >> dim;

    if (dim != 2)
    {
        throw std::logic_error("We only support meshes embedded in two dimensions");
    }

    nodeCoordinates.resize(2 * nNode);
    for (int iNode = 0; iNode < nNode; ++iNode)
    {
        file >> nodeCoordinates[2 * iNode] >> nodeCoordinates[2 * iNode + 1];
    }

    file >> nBGroup;

    nBFace.resize(nBGroup);
    nf.resize(nBGroup);
    title.resize(nBGroup);
    B2N.resize(nBGroup);

    for (int iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        file >> nBFace[iBGroup] >> nf[iBGroup] >> title[iBGroup];

        if (nf[iBGroup] != 2)
        {
            throw std::logic_error("We only support boundary faces with two nodes");
        }

        B2N[iBGroup].resize(2 * nBFace[iBGroup]);

        for (int iBFace = 0; iBFace < nBFace[iBGroup]; ++iBFace)
        {
            file >> B2N[iBGroup][2 * iBFace] >> B2N[iBGroup][2 * iBFace + 1];
        }
    }

    int nElem;

    file >> nElem >> order >> basis;

    if (nElemTot != nElem)
    {
        throw std::logic_error("We only support one group of elements");
    }

    if (basis.compare("TriLagrange") != 0)
    {
        throw std::logic_error("We only support TriLagrange elements");
    }

    if (order != 1)
    {
        throw std::logic_error("We only support first order elements");
    }

    E2N.resize(3 * nElemTot);

    for (int iElem = 0; iElem < nElemTot; ++iElem)
    {
        file >> E2N[3 * iElem] >> E2N[3 * iElem + 1] >> E2N[3 * iElem + 2];
    }
}

void Mesh2d::output(const std::string &gridFile) const
{
    std::ofstream file(gridFile);

    file << nNode << " " << nElemTot << " " << dim << std::endl;

    for (int iNode = 0; iNode < nNode; ++iNode)
    {
        file << std::scientific << std::setprecision(15)
             << nodeCoordinates[2 * iNode] << " "
             << std::scientific << std::setprecision(15)
             << nodeCoordinates[2 * iNode + 1]
             << std::endl;
    }

    file << nBGroup << std::endl;

    for (int iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        file << nBFace[iBGroup] << " " << nf[iBGroup] << " " << title[iBGroup]
             << std::endl;

        for (int iBFace = 0; iBFace < nBFace[iBGroup]; ++iBFace)
        {
            file << B2N[iBGroup][2 * iBFace]     << " "
                 << B2N[iBGroup][2 * iBFace + 1] << std::endl;
        }
    }

    file << nElemTot << " " << order << " " << basis << std::endl;

    for (int iElem = 0; iElem < nElemTot; ++iElem)
    {
        file << E2N[3 * iElem]     << " "
             << E2N[3 * iElem + 1] << " "
             << E2N[3 * iElem + 2] << std::endl;
    }
}
