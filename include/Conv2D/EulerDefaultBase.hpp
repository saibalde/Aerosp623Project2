#ifndef EULERDEFUALTBASE_HPP
#define EULERDEFAULTBASE_HPP

#include <string>

#include <armadillo>

#include "Conv2D/Mesh.hpp"

/**
 * @brief A class implementing Euler advection on 2D geometries
 */
class EulerDefaultBase
{
public:
    /**
     * @brief Default constructor
     */
    EulerDefaultBase() = default;

    /**
     * @brief Default destructor
     */
    virtual ~EulerDefaultBase() = default;

    /**
     * @brief Assign the unstructured, triangular mesh describing the geometry
     */
    void setMesh(const Mesh &mesh)
    {
        mesh_ = mesh;
    }

    /**
     * @brief Set up the specific heat ratio
     */
    void setSpecificHeatRatio(double gamma)
    {
        gamma_ = gamma;
    }

    /**
     * @brief Set up the gas constant
     */
    void setGasConstant(double R)
    {
        R_ = R;
    }

    /**
     * @brief Set up the free stream Mach number
     */
    void setFreeFlowMachNumber(double MInf)
    {
        MInf_ = MInf;
    }

    /**
     * @brief Set up the free stream static pressure
     */
    void setFreeFlowStaticPressure(double pInf)
    {
        pInf_ = pInf;
    }

    /**
     * @brief Set up the CFL number for timestepping
     */
    void setCFLNumber(double CFL)
    {
        CFL_ = CFL;
    }

    /**
     * @brief Set up the initial state to free stream state
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * gas constant, free stream mach number and free stream static pressure
     * has been already set for the class instance
     */
    void setInitialState();


    /**
     * @brief Set up the initial state from the values in file
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * gas constant, free stream mach number and free stream static pressure
     * has been already set for the class instance
     *
     * @param[in] File name containing the initial data
     */
    void setInitialState(const std::string fileName);

    /**
     * @brief Compute Roe flux
     *
     * This routine should not be called unless the specific heat ratio has
     * aready been set
     *
     * @param[in]  UL State on the left
     * @param[in]  UR State on the right
     * @param[in]  n  Normal pointing from left to right
     * @param[out] F  Computed numerical flux
     * @param[out] s  Computed maximum wave speed across interface
     */
    void computeRoeFlux(const arma::vec &UL, const arma::vec &UR,
                        const arma::rowvec &n, arma::vec &F, double &s) const;

    /**
     * @brief Specify how to compute flux at bottom boundary
     *
     * By default compute Roe flux w.r.t. the free stream state. However, this
     * virtual function can be overriden. Examples include: applyFreeStreamBC,
     * applyInvisidWallBC, applyInflowBC, applyOutflowBC.
     *
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from mesh
     * @param[out] U    Computed boundary state
     * @param[out] F    Computed numerical flux
     * @param[out] s    Computed maximum wave speed across interface
     */
    virtual void computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                                   arma::vec &U, arma::vec &F,
                                   double &s) const;

    /**
     * @brief Specify how to compute flux at right boundary
     *
     * By default compute Roe flux w.r.t. the free stream state. However, this
     * virtual function can be overriden. Examples include: applyFreeStreamBC,
     * applyInvisidWallBC, applyInflowBC, applyOutflowBC.
     *
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from mesh
     * @param[out] U    Computed boundary state
     * @param[out] F    Computed numerical flux
     * @param[out] s    Computed maximum wave speed across interface
     */
    virtual void computeRightFlux(const arma::vec &UInt,  const arma::rowvec &n,
                                  arma::vec &U, arma::vec &F,
                                  double &s) const;

    /**
     * @brief Specify how to compute flux at top boundary
     *
     * By default compute Roe flux w.r.t. the free stream state. However, this
     * virtual function can be overriden. Examples include: applyFreeStreamBC,
     * applyInvisidWallBC, applyInflowBC, applyOutflowBC.
     *
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from mesh
     * @param[out] U    Computed boundary state
     * @param[out] F    Computed numerical flux
     * @param[out] s    Computed maximum wave speed across interface
     */
    virtual void computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                                arma::vec &U, arma::vec &F,
                                double &s) const;

    /**
     * @brief Specify how to compute flux at left boundary
     *
     * By default compute Roe flux w.r.t. the free stream state. However, this
     * virtual function can be overriden. Examples include: applyFreeStreamBC,
     * applyInvisidWallBC, applyInflowBC, applyOutflowBC.
     *
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from mesh
     * @param[out] U    Computed boundary state
     * @param[out] F    Computed numerical flux
     * @param[out] s    Computed maximum wave speed across interface
     */
    virtual void computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                                 arma::vec &U, arma::vec &F,
                                 double &s) const;

    /**
     * @brief Run the first order solver for specified number of iterations
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * gas constant, free stream mach number and free stream static pressure
     * has been already set for the class instance, and an initial state has
     * been set
     *
     * @param[in] numIter      Number of iterations to run
     * @param[in] residualFile Name of the file to store the residuals
     */
    void runFirstOrderSolver(arma::uword numIter,
                             const std::string &residualFile);

    /**
     * @brief Run the first order solver until specified tolerance is met
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * gas constant, free stream mach number and free stream static pressure
     * has been already set for the class instance, and an initial state has
     * been set
     *
     * @param[in] tolerance    Tolerance level
     * @param[in] residualFile Name of the file to store the residuals
     */
    void runFirstOrderSolver(double tolerance,
                             const std::string &residualFile);

    /**
     * @brief Run the second order solver for specified number of iterations
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * gas constant, free stream mach number and free stream static pressure
     * has been already set for the class instance, and an initial state has
     * been set
     *
     * @param[in] numIter      Number of iterations to run
     * @param[in] residualFile Name of the file to store the residuals
     */
    void runSecondOrderSolver(arma::uword numIter,
                              const std::string &residualFile);

    /**
     * @brief Run the second order solver until specified tolerance is met
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * gas constant, free stream mach number and free stream static pressure
     * has been already set for the class instance, and an initial state has
     * been set
     *
     * @param[in] tolerance    Tolerance level
     * @param[in] residualFile Name of the file to store the residuals
     */
    void runSecondOrderSolver(double tolerance,
                              const std::string &residualFile);

    /**
     * @brief Write validation values to file
     *
     * The lift coefficent, drag coefficient and the entropy error are computed
     * and written to file
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * free stream mach number and free stream static pressure has been already
     * set for the class instance, and a state has been set (either through
     * specifying initial value or after running the solvers)
     *
     * @param[in] fileName Name of the output file
     */
    void writeValidationValuesTofile(const std::string &fileName) const;

    /**
     * @brief Write pressure coefficients to file
     *
     * Write the pressure coefficients with x coordinates of the midpoints
     * of the faces at the bottom boundary
     *
     * This routine should not be called unless the mesh, specific heat ratio,
     * free stream mach number and free stream static pressure has been already
     * set for the class instance, and a state has been set (either through
     * specifying initial value or after running the solvers)
     *
     * @param[in] fileName Name of the output file
     */
    void writePressureCoefficientsToFile(const std::string &fileName) const;

    /**
     * @brief Write Mach numbers to file
     *
     * Write the Mach numbers to a VTK file
     *
     * This routine should not be called unless the mesh and the state of
     * class instance is set (either through specifying initial value or after
     * running the solvers)
     *
     * @param[in] fileName Name of the output file
     */
    void writeMachNumbersToFile(const std::string &fileName) const;

    /**
     * @brief Write the state to file
     *
     * This routine should not be called unless some state has been set (either
     * through specifying initial value or after running the solvers)
     *
     * @param[in] fileName Name of the output file
     */
    void writeStateToFile(const std::string &fileName) const;

protected:
    /**
     * @brief Apply free stream boundary condition
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from the domain
     * @param[out] U    Constructed boundary state
     * @param[out] F    Computed numeric flux
     * @param[out] s    Computed maximum wavespeed
     */
    void applyFreeStreamBC(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &U, arma::vec &F, double &s) const;

    /**
     * @brief Apply inviscid boundary condition
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from the domain
     * @param[out] U    Constructed boundary state
     * @param[out] F    Computed numeric flux
     * @param[out] s    Computed maximum wavespeed
     */
    void applyInvisidWallBC(const arma::vec &UInt, const arma::rowvec &n,
                            arma::vec &U, arma::vec &F, double &s) const;

    /**
     * @brief Apply inflow boundary condition
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from the domain
     * @param[out] U    Constructed boundary state
     * @param[out] F    Computed numeric flux
     * @param[out] s    Computed maximum wavespeed
     */
    void applyInflowBC(const arma::vec &UInt, const arma::rowvec &n,
                       arma::vec &U, arma::vec &F, double &s) const;

    /**
     * @brief Apply outflow boundary condition
     * @param[in]  UInt Interior state
     * @param[in]  n    Normal pointing out from the domain
     * @param[out] U    Constructed boundary state
     * @param[out] F    Computed numeric flux
     * @param[out] s    Computed maximum wavespeed
     */
    void applyOutflowBC(const arma::vec &UInt, const arma::rowvec &n,
                        arma::vec &U, arma::vec &F, double &s) const;

private:
    /**
     * @brief The mesh specifying the geometry
     */
    Mesh mesh_;

    /**
     * @brief Specific heat ratio
     */
    double gamma_;

    /**
     * @brief Gas constant
     */
    double R_;

    /**
     * @brief Free stream Mach number
     */
    double MInf_;

    /**
     * @brief Free stream static pressure
     */
    double pInf_;

    /**
     * @brief CFL number
     */
    double CFL_;

    /**
     * @brief Total stagnation inflow pressure
     */
    double pt_;

    /**
     * @brief Total stagnation inflow temperature
     */
    double Tt_;

    /**
     * @brief Free stream state
     */
    arma::vec Ufree_;

    /**
     * @brief Full state
     */
    arma::mat U_;

    /**
     * @brief Compute the free stream state
     */
    void computeFreeStreamState();

    /**
     * @brief Compute first order residual
     * @param[in]  U Current state
     * @param[out] R Computed residual tallies
     * @param[out] S Computed wave speed tallies
     * @return Infinity norm of the residual
     */
    double computeFirstOrderResidual(const arma::mat &U, arma::mat &R,
                                     arma::vec &S) const;

    /**
     * @brief Compute state gradients
     * @param[in]  U Current state
     * @param[out] G Computed gradients
     */
    void computeGradients(const arma::mat &U, arma::mat &G) const;

    /**
     * @brief Compute second order residual
     * @param[in]  U Current state
     * @param[in]  G Current gradients
     * @param[out] R Computed residual tallies
     * @param[out] S Computed wave speed tallies
     * @return Infinity norm of the residual
     */
    double computeSecondOrderResidual(const arma::mat &U, const arma::mat &G,
                                      arma::mat &R, arma::vec &S) const;

    /**
     * @brief Compute lift coefficient
     * @return Value of the lift coefficient
     */
    double liftCoefficient() const;

    /**
     * @brief Compute drag coefficient
     * @return Value of the drag coefficient
     */
    double dragCoefficient() const;

    /**
     * @brief Compute entropy error
     * @return Value of the entropy error
     */
    double entropyError() const;
};

#endif // EULERDEFAULTBASE_HPP
