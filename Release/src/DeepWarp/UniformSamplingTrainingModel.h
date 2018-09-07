#pragma once
#include "WarpingMapTrainingModel.h"

class UniformSamplingTrainingModel:public WarpingMapTrainingModel
{
public:
	UniformSamplingTrainingModel(MatrixXd* subspaceModes_,
		int r, double timestep, SparseMatrix<double>* massMatrix_,
		LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef,bool useStaticSolver = true);
	~UniformSamplingTrainingModel();

	virtual void excute();

protected:

};

