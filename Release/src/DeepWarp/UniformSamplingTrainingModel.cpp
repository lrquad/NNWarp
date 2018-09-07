#include "UniformSamplingTrainingModel.h"

#include "Integrator/ImplicitNewMatkSparseIntegrator.h"
#include <random>
#include <fstream>
#include <iostream>
#include <time.h> 
UniformSamplingTrainingModel::UniformSamplingTrainingModel(MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver) :WarpingMapTrainingModel(subspaceModes_, r, timestep, massMatrix_, forcemodel_, nonlinearforceModel_, numConstrainedDOFs_, constrainedDOFs_, dampingMassCoef, dampingStiffnessCoef, useStaticSolver)
{


}

UniformSamplingTrainingModel::~UniformSamplingTrainingModel()
{

}

void UniformSamplingTrainingModel::excute()
{
	int numModes = subspaceModes->cols();
	int numSamplePerModes = getNumTrainingSet();
	numSamplePerModes = 2;

	//this is not possible
	int totalSample = pow(numSamplePerModes,numModes);

	double minForce = -this->forceScale;
	double maxForce = this->forceScale;
	maxForce = 2;
	minForce = -1;

	double interval_ = (maxForce - minForce) / ((double)numSamplePerModes + 1);


	trainingLinearDis.clear();
	trainingNonLinearDis.clear();
	int maxIteration = 1000;
	
	VectorXd modesScale(numModes);

	for (int i = 0; i < totalSample; i++)
	{
		for (int j = 0; j < numModes; j++)
		{
			int sampleindex = i % (int)std::pow(numSamplePerModes, j + 1);
			sampleindex /= std::pow(numSamplePerModes, j);

			double scale_ = (sampleindex+1)*interval_ - minForce;
			modesScale[j] = scale_;
		}

		VectorXd fullforce = (*subspaceModes)*modesScale;
		std::cout << modesScale.transpose() << std::endl;

		integrator->setExternalForces(fullforce.data());
		integrator->doTimeStep();
		VectorXd q_dis;
		q_dis = integrator->getVectorq();

		//store training linear dis
		trainingLinearDis.push_back(q_dis);

		//=============================

		nonLinearIntegrator->setExternalForces(fullforce.data());

		double preNorm = DBL_MAX;
		double residual;
		for (int j = 0; j < maxIteration; j++)
		{
			nonLinearIntegrator->doTimeStep();
			double norm_ = nonLinearIntegrator->getVectorq().norm();
			residual = std::abs((norm_ - preNorm) / preNorm);
			preNorm = norm_;
			if (residual < 1e-10)
			{
				break;
			}
		}
		std::cout << residual << std::endl;

		q_dis = nonLinearIntegrator->getVectorq();

		//store training nonlinear dis
		trainingNonLinearDis.push_back(q_dis);
	}

	std::cout << "excute finished" << std::endl;

}
