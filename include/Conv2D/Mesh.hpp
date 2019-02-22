#ifndef MESH_HPP
#define MESH_HPP

#include <string>

#include <armadillo>

/**
 * @brief Class for reading, writing .gri files and computing mesh matrices
 */
struct Mesh
{
    /**
     * @brief Read the mesh from file
     * @param[in] inFile Name of input .gri file name
     */
    void readFromFile(const std::string &inFile);

    /**
     * @brief Write mesh to file
     * @param[in] outFile Name of output file
     */
    void writeToFile(const std::string &outFile) const;

    /**
     * @brief Compute the mesh Matrices
     */
    void computeMatrices();

    /**
     * @brief Write Matrices to file
     * @param[in] matFile Name of output file
     */
    void writeMatricesToFile(const std::string &matFile) const;

    /**
     * @brief Number of nodes
     */
    arma::uword nNode;

    /**
     * @brief Number of elements
     */
    arma::uword nElemTot;

    /**
     * @brief Dimension of the space in which the mesh is embedded
     */
    arma::uword dim;

    /**
     * @brief Coordinates of the node points
     */
    arma::mat nodeCoordinates;

    /**
     * @brief Number of boundary groups
     */
    arma::uword nBGroup;

    /**
     * @brief Number of faces in each boundary group
     */
    arma::uvec nBFace;

    /**
     * @brief Number of boundary nodes per face in each boundary group
     */
    arma::uvec nf;

    /**
     * @brief Title of each boundary groups
     */
    arma::field<std::string> title;

    /**
     * @brief Mapping of boundary faces to node numbers
     */
    arma::field<arma::umat> B2N;

    /**
     * @brief Order of the elements
     */
    arma::uword order;

    /**
     * @brief Type of the elemeents
     */
    std::string basis;

    /**
     * @brief Mapping of elements to node numbers
     */
    arma::umat E2N;

    /**
     * @brief Mapping of interior faces to face midpoints
     */
    arma::mat  I2M;

    /**
     * @brief Mapping of interior faces to elements
     *
     * Each row contains four entries [elemL, faceL, elemR, faceR], where
     * elemL and elemR are the indices of the neighboring elements (we ensure
     * elemL < elemR), and faceL and faceR are the local node numbers of the
     * node opposite to the face.
     */
    arma::umat I2E;

    /**
     * @brief Mapping of interior faces to unit normals
     *
     * Unit normals point from elemL to elemR
     */
    arma::mat  In;

    /**
     * @brief Mapping of interior faces to face lengths
     */
    arma::vec  Il;

    /**
     * @brief Mapping of boundary faces to face midpoints
     */
    arma::mat  B2M;

    /**
     * @brief Mapping of boundary faces to elements
     *
     * Each row contains four entries [elem, face, bgroup], where elem is the
     * index of the adjoining element, and face is the local node numbers of
     * the node opposite to the face. Finally, bgroup is the number of the
     * group the boundary face belongs to.
     */
    arma::umat B2E;

    /**
     * @brief Mapping of boundary faces to unit normals
     *
     * Unit normals point out to the exterior
     */
    arma::mat  Bn;

    /**
     * @brief Mapping of boundary faces to face lengths
     */
    arma::vec  Bl;

    /**
     * @brief Mapping of elements to element midpoints
     */
    arma::mat E2M;

    /**
     * @brief Mapping of elements to element areas
     */
    arma::vec E2A;
};

#endif // MESH_HPP
