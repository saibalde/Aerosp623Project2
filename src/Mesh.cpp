#include "Conv2D/Mesh.hpp"

#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <cmath>

void Mesh::readFromFile(const std::string &inFile)
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

    nodeCoordinates.set_size(nNode, 2);
    for (arma::uword iNode = 0; iNode < nNode; ++iNode)
    {
        file >> nodeCoordinates(iNode, 0) >> nodeCoordinates(iNode, 1);
    }

    file >> nBGroup;

    nBFace.set_size(nBGroup);
    nf.set_size(nBGroup);
    title.set_size(nBGroup);
    B2N.set_size(nBGroup);

    for (arma::uword iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        file >> nBFace(iBGroup) >> nf(iBGroup) >> title(iBGroup);

        if (nf(iBGroup) != 2)
        {
            throw std::logic_error("We only support boundary faces with two nodes");
        }

        B2N(iBGroup).set_size(nBFace(iBGroup), 2);

        for (arma::uword iBFace = 0; iBFace < nBFace(iBGroup); ++iBFace)
        {
            file >> B2N(iBGroup)(iBFace, 0) >> B2N(iBGroup)(iBFace, 1);
        }
    }

    arma::uword nElem;

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

    E2N.set_size(nElemTot, 3);

    for (arma::uword iElem = 0; iElem < nElemTot; ++iElem)
    {
        file >> E2N(iElem, 0) >> E2N(iElem, 1) >> E2N(iElem, 2);
    }
}

void Mesh::writeToFile(const std::string &outFile) const
{
    std::ofstream file(outFile);

    file << nNode << " " << nElemTot << " " << dim << std::endl;

    for (arma::uword iNode = 0; iNode < nNode; ++iNode)
    {
        file << std::scientific << std::setprecision(15)
             << nodeCoordinates(iNode, 0) << " "
             << std::scientific << std::setprecision(15)
             << nodeCoordinates(iNode, 1)
             << std::endl;
    }

    file << nBGroup << std::endl;

    for (arma::uword iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        file << nBFace(iBGroup) << " " << nf(iBGroup) << " " << title(iBGroup)
             << std::endl;

        for (arma::uword iBFace = 0; iBFace < nBFace(iBGroup); ++iBFace)
        {
            file << B2N(iBGroup)(iBFace, 0) << " "
                 << B2N(iBGroup)(iBFace, 1) << std::endl;
        }
    }

    file << nElemTot << " " << order << " " << basis << std::endl;

    for (arma::uword iElem = 0; iElem < nElemTot; ++iElem)
    {
        file << E2N(iElem, 0) << " "
             << E2N(iElem, 1) << " "
             << E2N(iElem, 2) << std::endl;
    }
}

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

void Mesh::computeMatrices()
{
    // total number of boundary faces
    arma::uword nBFaceTot = 0;
    for (arma::uword iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        nBFaceTot += nBFace(iBGroup);
    }

    // number of arma::uworderior faces (overestimated)
    arma::uword nIFaceTot = static_cast<arma::uword>(std::ceil(1.5 * nElemTot));

    // allocate memory for mesh matrices
    I2E.set_size (nIFaceTot, 4);
    In.set_size  (nIFaceTot, 2);
    Il.set_size  (nIFaceTot);
    B2E.set_size (nBFaceTot, 3);
    Bn.set_size  (nBFaceTot, 2);
    Bl.set_size  (nBFaceTot);
    area.set_size(nElemTot);

    // hash matrices
    arma::SpMat<arma::uword> hashElem(nNode, nNode);
    arma::SpMat<arma::uword> hashFace(nNode, nNode);

    // Update I2E, In and Il
    arma::uword nIFace = 0;

    for (arma::uword iElem = 0; iElem < nElemTot; ++iElem)
    {
        arma::Col<arma::uword>::fixed<3> nodeNums;
        nodeNums[0] = E2N(iElem, 0);
        nodeNums[1] = E2N(iElem, 1);
        nodeNums[2] = E2N(iElem, 2);

        for (arma::uword edge = 0; edge < 3; ++edge)
        {
            arma::uword nodeLocNum1 = edge % 3;
            arma::uword nodeLocNum2 = (edge + 1) % 3;
            arma::uword nodeLocNum3 = 3 - nodeLocNum1 - nodeLocNum2;

            arma::uword nodeNum1 = nodeNums(nodeLocNum1) - 1;
            arma::uword nodeNum2 = nodeNums(nodeLocNum2) - 1;
            arma::uword nodeNum3 = nodeNums(nodeLocNum3) - 1;

            arma::uword nodeMin = std::min(nodeNum1, nodeNum2);
            arma::uword nodeMax = std::max(nodeNum1, nodeNum2);

            const arma::uword oldElem = hashElem(nodeMin, nodeMax);

            if (oldElem == 0)
            {
                // access for the first time
                hashElem(nodeMin, nodeMax) = iElem + 1;
                hashFace(nodeMin, nodeMax) = nodeLocNum3 + 1;
            }
            else if (0 < oldElem || oldElem < nElemTot + 1)
            {
                // access second time: definitly interior edge
                arma::uword elemL = oldElem;
                arma::uword faceL = hashFace(nodeMin, nodeMax);
                arma::uword elemR = iElem + 1;
                arma::uword faceR = nodeLocNum3 + 1;

                I2E(nIFace, 0) = elemL;
                I2E(nIFace, 1) = faceL;
                I2E(nIFace, 2) = elemR;
                I2E(nIFace, 3) = faceR;

                const double ax = nodeCoordinates(nodeNum1, 0);
                const double ay = nodeCoordinates(nodeNum1, 1);
                const double bx = nodeCoordinates(nodeNum2, 0);
                const double by = nodeCoordinates(nodeNum2, 1);
                const double cx = nodeCoordinates(nodeNum3, 0);
                const double cy = nodeCoordinates(nodeNum3, 1);

                const double l = std::hypot(ax - bx, ay - by);
                Il(nIFace) = l;

                double nx, ny;
                computeUnitDirection(ax, ay, bx, by, cx, cy, nx, ny);

                In(nIFace, 0) = nx;
                In(nIFace, 1) = ny;

                hashElem(nodeMin, nodeMax) = nElemTot + 1;

                nIFace += 1;
            }
            else
            {
                throw std::logic_error("Something is wrong with the mesh");
            }
        }
    }

    // clip I2E, In and Il to their appropriate sizes
    I2E.resize(nIFace, 4);
    In.resize (nIFace, 2);
    Il.resize (nIFace);

    // update B2E, Bn and Bl
    arma::uword nBEdge = 0;

    for (arma::uword iBGroup = 0; iBGroup < nBGroup; ++iBGroup)
    {
        for (arma::uword iBFace = 0; iBFace < nBFace(iBGroup); ++iBFace)
        {
            arma::uword nodeNum1 = B2N(iBGroup)(iBFace, 0) - 1;
            arma::uword nodeNum2 = B2N(iBGroup)(iBFace, 1) - 1;

            arma::uword nodeMin = std::min(nodeNum1, nodeNum2);
            arma::uword nodeMax = std::max(nodeNum1, nodeNum2);

            arma::uword elem = hashElem(nodeMin, nodeMax);
            arma::uword face = hashFace(nodeMin, nodeMax);

            B2E(nBEdge, 0) = elem;
            B2E(nBEdge, 1) = face;
            B2E(nBEdge, 2) = iBGroup + 1;

            arma::uword nodeNum3 = E2N(elem - 1, face - 1) - 1;

            const double ax = nodeCoordinates(nodeNum1, 0);
            const double ay = nodeCoordinates(nodeNum1, 1);
            const double bx = nodeCoordinates(nodeNum2, 0);
            const double by = nodeCoordinates(nodeNum2, 1);
            const double cx = nodeCoordinates(nodeNum3, 0);
            const double cy = nodeCoordinates(nodeNum3, 1);

            const double l = std::hypot(ax - bx, ay - by);
            Bl(nBEdge) = l;

            double nx, ny;
            computeUnitDirection(ax, ay, bx, by, cx, cy, nx, ny);

            Bn(nBEdge, 0) = -nx;
            Bn(nBEdge, 1) = -ny;

            nBEdge += 1;
        }
    }

    // update area
    for (arma::uword iElem = 0; iElem < nElemTot; ++iElem)
    {
        arma::uword nodeNum1 = E2N(iElem, 0) - 1;
        arma::uword nodeNum2 = E2N(iElem, 1) - 1;
        arma::uword nodeNum3 = E2N(iElem, 2) - 1;

        double ax = nodeCoordinates(nodeNum1, 0);
        double ay = nodeCoordinates(nodeNum1, 1);
        double bx = nodeCoordinates(nodeNum2, 0);
        double by = nodeCoordinates(nodeNum2, 1);
        double cx = nodeCoordinates(nodeNum3, 0);
        double cy = nodeCoordinates(nodeNum3, 1);

        area(iElem) = 0.5 * std::abs(ax * (by - cy) + bx * (cy - ay) +
                                     cx * (ay - by));
    }
}


void Mesh::writeMatricesToFile(const std::string &matFile) const
{
    std::ofstream file(matFile);

    arma::uword nIFace = In.n_rows;
    arma::uword nBFace = Bn.n_rows;

    file << "I2E" << std::endl << "===" << std::endl;

    for (arma::uword iIFace = 0; iIFace < nIFace; ++iIFace)
    {
        file << I2E(iIFace, 0) << " "
             << I2E(iIFace, 1) << " "
             << I2E(iIFace, 2) << " "
             << I2E(iIFace, 3) << std::endl;
    }

    file << std::endl << "B2E" << std::endl << "===" << std::endl;

    for (arma::uword iBFace = 0; iBFace < nBFace; ++iBFace)
    {
        file << B2E(iBFace, 0) << " "
             << B2E(iBFace, 1) << " "
             << B2E(iBFace, 2) << std::endl;
    }

    file << std::endl << "In" << std::endl << "==" << std::endl;

    for (arma::uword iIFace = 0; iIFace < nIFace; ++iIFace)
    {
        file << std::scientific << std::setprecision(15)
             << In(iIFace, 0) << " "
             << std::scientific << std::setprecision(15)
             << In(iIFace, 1) << std::endl;
    }

    file << std::endl << "Bn" << std::endl << "==" << std::endl;

    for (arma::uword iBFace = 0; iBFace < nBFace; ++iBFace)
    {
        file << std::scientific << std::setprecision(15)
             << Bn(iBFace, 0) << " "
             << std::scientific << std::setprecision(15)
             << Bn(iBFace, 1) << std::endl;
    }

    file << std::endl << "Il" << std::endl << "==" << std::endl;

    for (arma::uword iIFace = 0; iIFace < nIFace; ++iIFace)
    {
        file << std::scientific << std::setprecision(15)
             << Il(iIFace) << std::endl;
    }

    file << std::endl << "Bl" << std::endl << "==" << std::endl;

    for (arma::uword iBFace = 0; iBFace < nBFace; ++iBFace)
    {
        file << std::scientific << std::setprecision(15)
             << Bl(iBFace) << std::endl;
    }

    file << std::endl << "area" << std::endl << "====" << std::endl;

    for (arma::uword iElem = 0; iElem < nElemTot; ++iElem)
    {
        file << std::scientific << std::setprecision(15)
             << area(iElem) << std::endl;
    }
}
