#ifndef EULERDEFUALTBASE_HPP
#define EULERDEFAULTBASE_HPP

#include <string>

#include <armadillo>

#include "Conv2D/Mesh.hpp"

class EulerDefaultBase
{
public:
    EulerDefaultBase() : stepNum_(0)
    {
    }

    virtual ~EulerDefaultBase() = default;

    void setMesh(const Mesh &mesh)
    {
        mesh_ = mesh;
    }

    void setSpecificHeatRatio(double gamma)
    {
        gamma_ = gamma;
    }

    void setGasConstant(double R)
    {
        R_ = R;
    }

    void setFreeFlowMachNumber(double MInf)
    {
        MInf_ = MInf;
    }

    void setFreeFlowStaticPressure(double pInf)
    {
        pInf_ = pInf;
    }

    void setCFLNumber(double CFL)
    {
        CFL_ = CFL;
    }

    void setInitialState();

    void setInitialState(const std::string fileName);

    void computeRoeFlux(const arma::vec &UL, const arma::vec &UR,
                        const arma::rowvec &n, arma::vec &F, double &s) const;

    virtual void computeBottomFlux(const arma::vec &UInt, const arma::rowvec &n,
                                   arma::vec &U, arma::vec &F,
                                   double &s) const;

    virtual void computeRightFlux(const arma::vec &UInt,  const arma::rowvec &n,
                                  arma::vec &U, arma::vec &F,
                                  double &s) const;

    virtual void computeTopFlux(const arma::vec &UInt, const arma::rowvec &n,
                                arma::vec &U, arma::vec &F,
                                double &s) const;

    virtual void computeLeftFlux(const arma::vec &UInt, const arma::rowvec &n,
                                 arma::vec &U, arma::vec &F,
                                 double &s) const;

    void runFirstOrderSolver(arma::uword numIter,
                             const std::string &residualFile);

    void runFirstOrderSolver(double tolerance,
                             const std::string &residualFile);

    void runSecondOrderSolver(arma::uword numIter,
                              const std::string &residualFile);

    void runSecondOrderSolver(double tolerance,
                              const std::string &residualFile);

    void writeValidationValuesTofile(const std::string &fileName) const;

    void writePressureCoefficientsToFile(const std::string &fileName) const;

    void writeMachNumbersToFile(const std::string &fileName) const;

    void writeStateToFile(const std::string &fileName) const;

protected:
    void applyFreeStreamBC(const arma::vec &UInt, const arma::rowvec &n,
                           arma::vec &U, arma::vec &F, double &s) const;

    void applyInvisidWallBC(const arma::vec &UInt, const arma::rowvec &n,
                            arma::vec &U, arma::vec &F, double &s) const;

    void applyInflowBC(const arma::vec &UInt, const arma::rowvec &n,
                       arma::vec &U, arma::vec &F, double &s) const;

    void applyOutflowBC(const arma::vec &UInt, const arma::rowvec &n,
                        arma::vec &U, arma::vec &F, double &s) const;

private:
    arma::uword stepNum_;

    Mesh mesh_;

    double gamma_;
    double R_;
    double MInf_;
    double pInf_;
    double CFL_;

    double pt_;
    double Tt_;

    arma::vec Ufree_;
    arma::mat U_;

    void computeFreeStreamState();

    double computeFirstOrderResidual(const arma::mat &U, arma::mat &R,
                                     arma::vec &S) const;

    void computeGradients(const arma::mat &U, arma::mat &G) const;

    double computeSecondOrderResidual(const arma::mat &U, const arma::mat &G,
                                      arma::mat &R, arma::vec &S) const;

    double liftCoefficient() const;

    double dragCoefficient() const;

    double entropyError() const;
};

#endif // EULERDEFAULTBASE_HPP
