#include "SmartTrainingModel.h"
#include "Integrator/ImplicitNewMatkSparseIntegrator.h"
#include <random>
#include <fstream>
#include <iostream>
#include <time.h> 
#include "Simulator/DeepWarp/ModalRotationMatrix.h"
#include "Functions/GeoMatrix.h"

SmartTrainingModel::SmartTrainingModel(VectorXd gravity_, MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver /*= true*/) :WarpingMapTrainingModel(subspaceModes_, r, timestep, massMatrix_, forcemodel_, nonlinearforceModel_, numConstrainedDOFs_, constrainedDOFs_, dampingMassCoef, dampingStiffnessCoef, useStaticSolver)
{
	integrator->setDampingMassCoef(0.1);
	integrator->setDampingStiffnessCoef(0.1);
	nonLinearIntegrator->setDampingMassCoef(0.1);
	nonLinearIntegrator->setDampingStiffnessCoef(0.1);
	this->gravity = gravity_;
}

SmartTrainingModel::~SmartTrainingModel()
{

}

void SmartTrainingModel::excute()
{
	std::cout << "num low fre data." << getNumTrainingSet() << std::endl;
	std::cout << "num high fre data." << getNumTrainingHighFreq() << std::endl;

	VectorXd reducedForce(subspaceModes->cols());
	trainingLinearDis.clear();
	trainingNonLinearDis.clear();

	int maxIteration = 1000;

	trainingLinearDis.reserve(getNumTrainingSet() + getNumTrainingHighFreq());
	trainingNonLinearDis.reserve(getNumTrainingSet() + getNumTrainingHighFreq());

	int dataCount = 0;
	while (dataCount < getNumTrainingSet())
	{
		std::cout << dataCount << std::endl;
		srand(time(NULL) + dataCount);
		std::default_random_engine generator;
		generator.seed(time(NULL) + dataCount);
		std::uniform_real_distribution<double> latitude_distribution(-forceScale, forceScale);
		double scale = (this->forceScale / getNumTrainingSet()) *(getNumTrainingSet() - dataCount) + 0.01;
		double angle = (PI_ / 4 / getNumTrainingSet())*dataCount - PI_ / 8 + 0.1;

		Vector3d axis(0, 0, 1);

		VectorXd fullforce = this->rotationGravity(gravity, angle, axis);

		fullforce = gravity*scale;

		/*fullforce.setRandom();
		fullforce.normalize();
		fullforce *= scale;*/
		
		nonLinearIntegrator->setExternalForces(fullforce.data());
		nonLinearIntegrator->resetToRest();
		nonLinearIntegrator->setSaveStepResidual(true);
		VectorXd q_dis;
		
		//store training linear dis
		//trainingLinearDis.push_back(q_dis);
		double preNorm = DBL_MAX;
		VectorXd preDis = gravity;
		preDis.setConstant(1);

		double residual;

		for (int j = 0; j < maxIteration; j++)
		{
			nonLinearIntegrator->doTimeStep();
			double norm_ = nonLinearIntegrator->getVectorq().norm();
			VectorXd cur_dis = nonLinearIntegrator->getVectorq();

			residual = std::abs((preNorm - norm_) / preNorm);
			//We only pick the displacement which has larget difference
			if (residual > 0.001)
			{
				if (norm_ > 0.01)
				{
					trainingNonLinearDis.push_back(cur_dis);

					VectorXd curExternalForce = nonLinearIntegrator->getStep_residual();

					integrator->setExternalForces(curExternalForce.data());
					integrator->doTimeStep();

					q_dis = integrator->getVectorq();

					trainingLinearDis.push_back(q_dis);

					dataCount++;
				}
			}
			else
			{
				break;
			}

			preNorm = norm_;
			if (residual < 1e-5)
			{
				break;
			}
		}

		nonLinearIntegrator->setSaveStepResidual(false);
	}

	std::cout << "excute finished" << std::endl;
}

void SmartTrainingModel::convergeIntegrationNonLinear_(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration)
{
	integrator_loc->resetToRest();
	std::cout << "convergeIntegrationNonLinear === >" << std::endl;
	double preNorm = DBL_MAX;
	double residual = DBL_MAX;

	integrator_loc->setExternalForces(extForce.data());
	for (int j = 0; j < maxIteration; j++)
	{
		integrator_loc->doTimeStep();
		double norm_ = integrator_loc->getVectorq().norm();
		residual = std::abs((norm_ - preNorm) / preNorm);
		preNorm = norm_;
		if (residual < 1e-10)
		{
			break;
		}
	}
	std::cout << residual << std::endl;
}

