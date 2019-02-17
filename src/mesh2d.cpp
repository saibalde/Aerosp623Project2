#include "mesh2d.hpp"

#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <cmath>

#include <armadillo>

void computeUnitDirection(double  ax, double  ay, double  bx, double  by,
                          double  cx, double  cy, double &nx, double &ny)
{
    double abx = bx - ax;
    double aby = by - ay;

    double ab = std::hypot(abx, aby);

    abx /= ab;
    aby /= ab;

    double acx = cx - ax;
    double acy = cy - ay;

    double dot = abx * acx + aby * acy;

    acx -= dot * abx;
    acy -= dot * aby;

    double ac = std::hypot(acx, acy);

    nx = acx / ac;
    ny = acy / ac;
}

Mesh2d::Mesh2d(const std::string &inFile)
{
    std::ifstream file(inFile);

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

void Mesh2d::setupMatrices()
{
    // total number of boundary faces
    int nBFaceTot = 0;
    for (int iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        nBFaceTot += nBFace[iBGroup];
    }

    // number of interior faces (overestimated)
    int nIFaceTot = static_cast<int>(std::ceil(1.5 * nElemTot));

    // allocate memory for mesh matrices
    I2E.resize(4 * nIFaceTot);
    B2E.resize(3 * nBFaceTot);
    In.resize(2 * nIFaceTot);
    Bn.resize(2 * nBFaceTot);
    area.resize(nElemTot);

    // hash matrices
    arma::SpMat<int> hashElem(nNode, nNode);
    arma::SpMat<int> hashFace(nNode, nNode);

    // Update I2E and In
    int nIFace = 0;

    for (int iElem = 0; iElem < nElemTot; ++iElem)
    {
        std::vector<int> nodeNums(3);
        nodeNums[0] = E2N[3 * iElem];
        nodeNums[1] = E2N[3 * iElem + 1];
        nodeNums[2] = E2N[3 * iElem + 2];

        for (int edge = 0; edge < 3; ++edge)
        {
            int nodeLocNum1 = edge % 3;
            int nodeLocNum2 = (edge + 1) % 3;
            int nodeLocNum3 = 3 - nodeLocNum1 - nodeLocNum2;

            int nodeNum1 = nodeNums[nodeLocNum1] - 1;
            int nodeNum2 = nodeNums[nodeLocNum2] - 1;
            int nodeNum3 = nodeNums[nodeLocNum3] - 1;

            int nodeMin = std::min(nodeNum1, nodeNum2);
            int nodeMax = std::max(nodeNum1, nodeNum2);

            if (hashElem(nodeMin, nodeMax) == 0)
            {
                // access for the first time
                hashElem(nodeMin, nodeMax) = iElem + 1;
                hashFace(nodeMin, nodeMax) = nodeLocNum3 + 1;
            }
            else if (hashElem(nodeMin, nodeMax) > 0)
            {
                // access second time: definitly interior edge
                int oldElem = hashElem(nodeMin, nodeMax);

                int elemL = oldElem;
                int faceL = hashFace(nodeMin, nodeMax);
                int elemR = iElem + 1;
                int faceR = nodeLocNum3 + 1;

                I2E[4 * nIFace]     = elemL;
                I2E[4 * nIFace + 1] = faceL;
                I2E[4 * nIFace + 2] = elemR;
                I2E[4 * nIFace + 3] = faceR;

                double nx, ny;
                computeUnitDirection(nodeCoordinates[2 * nodeNum1],
                                     nodeCoordinates[2 * nodeNum1 + 1],
                                     nodeCoordinates[2 * nodeNum2],
                                     nodeCoordinates[2 * nodeNum2 + 1],
                                     nodeCoordinates[2 * nodeNum3],
                                     nodeCoordinates[2 * nodeNum3 + 1],
                                     nx, ny);

                In[2 * nIFace]     = nx;
                In[2 * nIFace + 1] = ny;

                hashElem(nodeMin, nodeMax) = -1;

                nIFace += 1;
            }
            else
            {
                throw std::logic_error("Something is wrong with the mesh");
            }
        }
    }

    I2E.resize(4 * nIFace);
    In.resize(2 * nIFace);

    // update B2E and Bn
    int nBEdge = 0;

    for (int iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        for (int iBFace = 0; iBFace < nBFace[iBGroup]; ++iBFace)
        {
            int nodeNum1 = B2N[iBGroup][2 * iBFace]     - 1;
            int nodeNum2 = B2N[iBGroup][2 * iBFace + 1] - 1;


            int nodeMin = std::min(nodeNum1, nodeNum2);
            int nodeMax = std::max(nodeNum1, nodeNum2);

            int elem = hashElem(nodeMin, nodeMax);
            int face = hashFace(nodeMin, nodeMax);

            B2E[3 * nBEdge]     = elem;
            B2E[3 * nBEdge + 1] = face;
            B2E[3 * nBEdge + 2] = iBGroup + 1;

            int nodeNum3 = E2N[3 * (elem - 1) + (face - 1)] - 1;

            double nx, ny;
            computeUnitDirection(nodeCoordinates[2 * nodeNum1],
                                 nodeCoordinates[2 * nodeNum1 + 1],
                                 nodeCoordinates[2 * nodeNum2],
                                 nodeCoordinates[2 * nodeNum2 + 1],
                                 nodeCoordinates[2 * nodeNum3],
                                 nodeCoordinates[2 * nodeNum3 + 1],
                                 nx, ny);

            Bn[2 * nBEdge]     = -nx;
            Bn[2 * nBEdge + 1] = -ny;

            nBEdge += 1;
        }
    }

    // update area
    for (int iElem = 0; iElem < nElemTot; ++iElem)
    {
        int nodeNum1 = E2N[3 * iElem]     - 1;
        int nodeNum2 = E2N[3 * iElem + 1] - 1;
        int nodeNum3 = E2N[3 * iElem + 2] - 1;

        double ax = nodeCoordinates[2 * nodeNum1];
        double ay = nodeCoordinates[2 * nodeNum1 + 1];
        double bx = nodeCoordinates[2 * nodeNum2];
        double by = nodeCoordinates[2 * nodeNum2 + 1];
        double cx = nodeCoordinates[2 * nodeNum3];
        double cy = nodeCoordinates[2 * nodeNum3 + 1];

        area[iElem] = 0.5 * std::abs(ax * (by - cy) + bx * (cy - ay) +
                                     cx * (ay - by));
    }
}

void Mesh2d::output(const std::string &outFile) const
{
    std::ofstream file(outFile);

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

void Mesh2d::outputMatrices(const std::string &matFile) const
{
    std::ofstream file(matFile);

    int nIFace = In.size() / 2;
    int nBFace = Bn.size() / 2;

    file << "I2E" << std::endl << "===" << std::endl;

    for (int iIFace = 0; iIFace < nIFace; ++iIFace)
    {
        file << I2E[4 * iIFace]     << " "
             << I2E[4 * iIFace + 1] << " "
             << I2E[4 * iIFace + 2] << " "
             << I2E[4 * iIFace + 3] << std::endl;
    }

    file << std::endl << "B2E" << std::endl << "===" << std::endl;

    for (int iBFace = 0; iBFace < nBFace; ++iBFace)
    {
        file << B2E[3 * iBFace]     << " "
             << B2E[3 * iBFace + 1] << " "
             << B2E[3 * iBFace + 2] << std::endl;
    }

    file << std::endl << "In" << std::endl << "==" << std::endl;

    for (int iIFace = 0; iIFace < nIFace; ++iIFace)
    {
        file << std::scientific << std::setprecision(15)
             << In[2 * iIFace]
             << std::scientific << std::setprecision(15)
             << " " << In[2 * iIFace + 1] << std::endl;
    }

    file << std::endl << "Bn" << std::endl << "==" << std::endl;

    for (int iBFace = 0; iBFace < nBFace; ++iBFace)
    {
        file << std::scientific << std::setprecision(15)
             << Bn[2 * iBFace] << " "
             << std::scientific << std::setprecision(15)
             << Bn[2 * iBFace + 1] << std::endl;
    }

    file << std::endl << "area" << std::endl << "====" << std::endl;

    for (int iElem = 0; iElem < nElemTot; ++iElem)
    {
        file << std::scientific << std::setprecision(15)
             << area[iElem] << std::endl;
    }
}
