#pragma once
#include "WarpingMapTrainingModel.h"
#include <Eigen/SparseQR>  

class LoboIntegrator;
class ModalRotationMatrix;
class LoboVolumetricMesh;
class LoboNodeBase;
class ImplicitModalWarpingIntegrator;
class RotateForceField;
class ReducedSTVKModel;
class ReducedForceModel;
class ImpicitNewMarkDenseIntegrator;

class ModalWarpingTrainingModel :public WarpingMapTrainingModel
{
public:
	ModalWarpingTrainingModel(LoboVolumetricMesh* volumtrciMesh_, VectorXd gravity_, SparseMatrix<double>* modalRotationSparseMatrix_, ModalRotationMatrix* modalrotationMatrix_, MatrixXd* subspaceModes_,
		int r, double timestep, SparseMatrix<double>* massMatrix_,
		LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver = true);
	~ModalWarpingTrainingModel();

	virtual void excute();

	virtual void getTraingLinearForce(int index, VectorXd &force);
	virtual void getTraingLinearForce(VectorXd& lq, VectorXd &force);
	virtual void getTraingLinearForce(VectorXd& lq, VectorXd &force, double poisson);
	virtual VectorXd getNonlinearInternalforce(int disid);

protected:

	virtual void subexcute3();
	virtual void subexcute();
	virtual void subexcute2();
	virtual void subexcute4();
	virtual void subexcute5();
	virtual void method6();
	virtual void method7();
	virtual void method8();
	virtual void method9();
	virtual void method10();


	virtual void method10Twist();
	virtual void method11Twist();
	virtual void method12Twist();
	virtual void methodFortwist();

	virtual bool convergeIntegrationLocal(LoboIntegrator* integrator_loc, VectorXd &extForce,int maxIteration,bool reset = true);
	virtual bool convergeIntegrationLocalBuffer(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, std::vector<VectorXd> &qn_list, bool reset = true);
	virtual bool convergeIntegrationLocalBuffer(LoboIntegrator* integrator_loc, RotateForceField* forcefiled, int maxIteration, std::vector<VectorXd> &qn_list, bool reset = true);

	virtual bool convergeIntegrationNonLinear(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset = true);

	virtual bool convergeIntegrationNonlinearReduced(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset = true);

	virtual void convergeIntegrationNonLinearBuffer(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration,std::vector<VectorXd> &qn_list, bool reset = true);

	virtual void convergeIntegrationNonLinearBuffer(LoboIntegrator* integrator_loc, RotateForceField* forcefiled, int maxIteration, std::vector<VectorXd> &qn_list, bool reset = true);

	void generateConstrainMatrixAndVector(SparseMatrix<double>* constrainMatrix, VectorXd &constrainTarget, LoboNodeBase* nodep,int nodeid ,VectorXd target_);

	double getMaxRotatedAngleFromW(VectorXd &w);


	virtual void samplePoissonRatio(double maxPoisson, double minPoisson, int maxNumPoisson, int minNumPoisson, std::vector<Vector3d> &nodeforce, std::vector<Vector3d> &forcedirection, std::vector<bool> &ifreset, std::vector<double> &poissonPerdirection);

	ImplicitModalWarpingIntegrator* modalwarpingintegrator;
	ImpicitNewMarkDenseIntegrator* reducedIntergrator;


	SparseMatrix<double>* modalRotationSparseMatrix;
	SparseMatrix<double>* localOrientationMatrixR;
	SparseMatrix<double>* matrixR_;
	ModalRotationMatrix* modalrotationMatrix;
	LoboVolumetricMesh* volumtrciMesh;
	VectorXd gravity;


	ReducedSTVKModel* reducedSTVKmodel;
	ReducedForceModel* reducedforcemodel;
	MatrixXd * reducedMassMatrix;

};

