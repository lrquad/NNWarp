#include "LinearMainSmartTrainingModel.h"
#include "Integrator/ImplicitNewMatkSparseIntegrator.h"
#include <random>
#include <fstream>
#include <iostream>
#include <time.h> 
#include "Functions/GeoMatrix.h"

LinearMainSmartTrainingModel::LinearMainSmartTrainingModel(VectorXd gravity_, MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver /*= false*/) :WarpingMapTrainingModel(subspaceModes_, r, timestep, massMatrix_, forcemodel_, nonlinearforceModel_, numConstrainedDOFs_, constrainedDOFs_, dampingMassCoef, dampingStiffnessCoef, useStaticSolver)
{
	integrator->setDampingMassCoef(0.1);
	integrator->setDampingStiffnessCoef(0.1);
	nonLinearIntegrator->setDampingMassCoef(1.0);
	nonLinearIntegrator->setDampingStiffnessCoef(1.0);
	this->gravity = gravity_;
}

LinearMainSmartTrainingModel::~LinearMainSmartTrainingModel()
{

}

void LinearMainSmartTrainingModel::excute()
{
	/*subExcute();
	return;*/

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
		double scale = (this->forceScale / getNumTrainingSet()) *(dataCount + 1) + 0.1;
		
		double angle = (PI_ / getNumTrainingSet())*dataCount- PI_/2 + 0.1;

		Vector3d axis(0, 0, 1);
		axis.setRandom();
		axis.normalize();

		VectorXd fullforce = this->rotationGravity(gravity, angle, axis);

		//fullforce.setRandom();
		//fullforce *= scale;

		fullforce = gravity*scale;

		/*fullforce.setZero();
		srand((unsigned int)time(0) + dataCount);

		axis.setRandom();
		axis.normalize();
		std::cout << "direction" << std::endl;
		std::cout << axis.transpose() << std::endl;

		fullforce.data()[0 * 3 + 0] = axis.data()[0];
		fullforce.data()[0 * 3 + 1] = axis.data()[1];
		fullforce.data()[0 * 3 + 2] = axis.data()[2];

		double scale_t = gravity.norm() / fullforce.norm();
		fullforce *= scale_t * 4;
		*/

		/*reducedForce.setRandom();
		reducedForce *= scale;
		fullforce = (*subspaceModes)*reducedForce;
		fullforce.setRandom();
		fullforce *= scale;*/

		integrator->setExternalForces(fullforce.data());
		integrator->resetToRest();
		integrator->setSaveStepResidual(true);

		double preNorm = DBL_MAX;
		double residual;
		for (int j = 0; j < maxIteration; j++)
		{
			integrator->doTimeStep();
			VectorXd cur_dis = integrator->getVectorq();

			double norm_ = cur_dis.norm();
			
			residual = std::abs((preNorm - norm_) / preNorm);
			preNorm = norm_;
			if (residual > 0.01)
			{
				if (norm_ > 0.01)
				{
					bool reset = false;
					if (j == 0)
					{
						reset = true;
					}

					trainingLinearDis.push_back(cur_dis);
					VectorXd curExternalForce = integrator->getStep_residual();
					convergeIntegrationNonLinear(nonLinearIntegrator, curExternalForce, maxIteration, reset);
					VectorXd q_dis = nonLinearIntegrator->getVectorq();
					trainingNonLinearDis.push_back(q_dis);
					dataCount++;
				}
			}
			else
			{
				break;
			}

		}
		integrator->setSaveStepResidual(false);
	}

	dataCount = 0;
	while (dataCount < getNumTrainingHighFreq())
	{
		std::cout << dataCount << std::endl;
		srand(time(NULL) + dataCount);
		std::default_random_engine generator;
		generator.seed(time(NULL) + dataCount);
		std::uniform_real_distribution<double> latitude_distribution(-forceScale, forceScale);
		double scale = (this->forceScale / getNumTrainingSet()) *(dataCount + 1) + 0.1;

		VectorXd fullforce = gravity*scale;
		fullforce.setRandom();
		fullforce *= scale;

		integrator->setExternalForces(fullforce.data());
		integrator->resetToRest();
		integrator->setSaveStepResidual(true);

		double preNorm = DBL_MAX;
		double residual;
		for (int j = 0; j < maxIteration; j++)
		{
			integrator->doTimeStep();
			VectorXd cur_dis = integrator->getVectorq();

			double norm_ = cur_dis.norm();

			residual = std::abs((preNorm - norm_) / preNorm);
			preNorm = norm_;
			if (residual > 0.01)
			{
				if (norm_ > 0.01)
				{
					trainingLinearDis.push_back(cur_dis);
					VectorXd curExternalForce = integrator->getStep_residual();
					bool reset = false;
					if (j == 0)
					{
						reset = true;
					}
					convergeIntegrationNonLinear(nonLinearIntegrator, curExternalForce, maxIteration, reset);
					VectorXd q_dis = nonLinearIntegrator->getVectorq();
					trainingNonLinearDis.push_back(q_dis);
					dataCount++;
				}
			}
			else
			{
				break;
			}

		}
		integrator->setSaveStepResidual(false);
	}

	std::cout << "excute finished" << std::endl;

}



void LinearMainSmartTrainingModel::subExcute()
{
	std::vector<Vector3d> nodedisplacement;
	
	generateRotationVectorSample(M_PI * 2, 0, 20,
		M_PI / 2, -M_PI / 2, 10,
		0, 0.4, 15, nodedisplacement);

	//nodedisplacement.clear();
	//for (int i = 0; i < 100; i++)
	//{
		//nodedisplacement.push_back(Vector3d(1, 0, 0)*(i-50)*0.01+Vector3d(0.001,0,0));
	//}

	int constrainedNodeid = 0;

	SparseMatrix<double> constrainMatrix;
	VectorXd constrainTarget;
	integrator->setStoreLagrangeMultipliers(true);

	this->setNumTrainingSet(nodedisplacement.size());
	this->setNumTrainingHighFreq(0);

	std::cout << "num low fre data." << getNumTrainingSet() << std::endl;
	std::cout << "num high fre data." << getNumTrainingHighFreq() << std::endl;

	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		std::cout << nodedisplacement[i].transpose() << std::endl;
		std::cout << i << std::endl;
		generateConstrainMatrixAndVector(&constrainMatrix, constrainTarget, constrainedNodeid, nodedisplacement[i]);
		integrator->setConstrainByLagrangeMulti(&constrainMatrix, constrainTarget);
		integrator->setExternalForcesToZero();
		integrator->doTimeStep();

		VectorXd multipliers = ((ImplicitNewMatkSparseIntegrator*)integrator)->getStoredLagrangeMultipliers();
		VectorXd constrainForce = -constrainMatrix.transpose()*multipliers;

		VectorXd cur_dis = integrator->getVectorq();
		trainingLinearDis.push_back(cur_dis);

		convergeIntegrationNonLinear(nonLinearIntegrator, constrainForce, 1000, true);
		VectorXd q_dis = nonLinearIntegrator->getVectorq();
		trainingNonLinearDis.push_back(q_dis);
	}

	std::cout << "finished" << std::endl;

}

void LinearMainSmartTrainingModel::convergeIntegrationLocal(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration)
{
	integrator_loc->resetToRest();
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
}

void LinearMainSmartTrainingModel::convergeIntegrationNonLinear(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset)
{
	//start from last step
	if (reset)
	integrator_loc->resetToRest();
	
	double preNorm = DBL_MAX;
	double residual = DBL_MAX;

	integrator_loc->setExternalForces(extForce.data());
	for (int j = 0; j < maxIteration; j++)
	{
		integrator_loc->doTimeStep();
		double norm_ = integrator_loc->getVectorq().norm();
		residual = std::abs((norm_ - preNorm) / preNorm);
		preNorm = norm_;
		if (residual < 1e-5)
		{
			break;
		}
	}
	std::cout << residual << std::endl;
}

typedef Eigen::Triplet<double> EIGEN_TRI;

void LinearMainSmartTrainingModel::generateConstrainMatrixAndVector(SparseMatrix<double>* constrainMatrix, VectorXd &constrainTarget, int nodeid, Vector3d target)
{
	int numVertex = this->gravity.rows();
	int totalConstrain = numConstrainedDOFs + 3;
	constrainMatrix->resize(totalConstrain, numVertex * 3);
	constrainTarget.resize(totalConstrain, 1);
	std::vector<EIGEN_TRI> entrys;

	for (int i = 0; i < numConstrainedDOFs; i++)
	{
		int row = i;
		int col = constrainedDOFs[i];
		entrys.push_back(EIGEN_TRI(row, col, 1));
		constrainTarget.data()[row] = 0;
	}

	for (int i = 0; i < 3; i++)
	{
		int row = i + numConstrainedDOFs;
		int col = nodeid * 3 + i;
		entrys.push_back(EIGEN_TRI(row, col, 1));
		constrainTarget.data()[row] = target.data()[i];
	}

	constrainMatrix->setFromTriplets(entrys.begin(), entrys.end());
}
