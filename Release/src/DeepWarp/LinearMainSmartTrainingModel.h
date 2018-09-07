#pragma once
#include "WarpingMapTrainingModel.h"

class LoboIntegrator;

class LinearMainSmartTrainingModel:public WarpingMapTrainingModel
{
public:
	LinearMainSmartTrainingModel(VectorXd gravity_, MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver = false);
	~LinearMainSmartTrainingModel();

	virtual void excute();

protected:

	virtual void subExcute();

	virtual void convergeIntegrationLocal(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration);
	virtual void convergeIntegrationNonLinear(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset);

	virtual void generateConstrainMatrixAndVector(SparseMatrix<double>* constrainMatrix, VectorXd &constrainTarget, int nodeid, Vector3d target);

	VectorXd gravity;

};

