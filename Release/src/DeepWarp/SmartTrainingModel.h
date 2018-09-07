#pragma once
#include "WarpingMapTrainingModel.h"

class LoboIntegrator;

class SmartTrainingModel:public WarpingMapTrainingModel
{
public:
	SmartTrainingModel(VectorXd gravity_, MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver = true);
	~SmartTrainingModel();

	virtual void excute();

protected:

	virtual void convergeIntegrationNonLinear_(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration);


	VectorXd gravity;
};

