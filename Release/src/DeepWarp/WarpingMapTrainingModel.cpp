#include "WarpingMapTrainingModel.h"
#include "Integrator/ImplicitNewMatkSparseIntegrator.h"
#include <random>
#include <fstream>
#include <iostream>
#include <time.h> 

#include "Functions/cnpy.h"

WarpingMapTrainingModel::WarpingMapTrainingModel(MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver)
{
	alreadyReadData = false;
	linearDisOnly = false;

	this->subspaceModes = subspaceModes_;
	this->loboforceModel = forcemodel_;
	this->nonlinearforceModel = nonlinearforceModel_;
	this->r = r;

	forceScale = 1;
	this->massMatrix = massMatrix_;
	integrator = new ImplicitNewMatkSparseIntegrator(
		r, timestep, massMatrix_, loboforceModel, 1, numConstrainedDOFs_, constrainedDOFs_, dampingMassCoef, dampingStiffnessCoef, 1, 1e-7, 0.25, 0.5, useStaticSolver
		);

	nonLinearIntegrator = new ImplicitNewMatkSparseIntegrator(
		r, timestep, massMatrix_, nonlinearforceModel_, 1, numConstrainedDOFs_, constrainedDOFs_, 0.4, 0.4, 100, 1e-7, 0.25, 0.5, false
		);

	setNumTrainingSet(100);
	setNumTestSet(100);
	setNumTrainingHighFreq(100);
	setNumTestHighFreq(100);

	this->numConstrainedDOFs = numConstrainedDOFs_;
	this->constrainedDOFs = constrainedDOFs_;
	forcefieldType = 0;
}

WarpingMapTrainingModel::~WarpingMapTrainingModel()
{
	delete integrator;
	delete nonLinearIntegrator;
}

void WarpingMapTrainingModel::excute()
{
	/*trainingLinearDis.resize(numTrainingSet);
	trainingNonLinearDis.resize(numTrainingSet);

	testLinearDis.resize(numTestSet);
	testNonLinearDis.resize(numTestSet);
	*/
	std::cout << "num low fre data." << getNumTrainingSet() << std::endl;
	std::cout << "num high fre data." << getNumTrainingHighFreq() << std::endl;

	VectorXd reducedForce(subspaceModes->cols());
	trainingLinearDis.clear();
	trainingNonLinearDis.clear();
	int maxIteration = 1000;

	for (int i = 0; i < getNumTrainingSet(); i++)
	{
		srand(time(NULL) + i);
		reducedForce.setRandom();
		reducedForce *= this->forceScale;
		VectorXd fullforce = (*subspaceModes)*reducedForce;

		integrator->setExternalForces(fullforce.data());
		integrator->doTimeStep();
		VectorXd q_dis;
		q_dis = integrator->getVectorq();

		//store training linear dis
		trainingLinearDis.push_back(q_dis);

		//=============================

		nonLinearIntegrator->setExternalForces(fullforce.data());
		nonLinearIntegrator->resetToRest();
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

	//generate high frequency force
	for (int i = 0; i < getNumTrainingHighFreq(); i++)
	{
		srand(time(NULL) + i);
		VectorXd fullforce(r);
		fullforce.setRandom();
		fullforce *= forceScale;
		integrator->setExternalForces(fullforce.data());
		integrator->doTimeStep();

		VectorXd q_dis;
		q_dis = integrator->getVectorq();

		//store high frequency training linear dis
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

void WarpingMapTrainingModel::exportExampleData(const char* filename, double* features, int dimension, int totalsize)
{
	//std::ofstream outputstream(filename);
	//int num_set = totalsize / dimension;
	//for (int i = 0; i < num_set; i++)
	//{
	//	for (int j = 0; j < dimension; j++)
	//	{
	//		outputstream << features[i*dimension + j] << ",";
	//	}
	//	outputstream << std::endl;
	//}
	//outputstream.close();

	size_t row = totalsize / dimension;
	size_t col = dimension;

	cnpy::npy_save(filename, features, {row,col}, "w");

}

void WarpingMapTrainingModel::exportExampleDataAscii(const char* filename, double* features, int dimension, int totalsize)
{
	std::ofstream outputstream(filename);
	int num_set = totalsize / dimension;
	for (int i = 0; i < num_set; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			outputstream << features[i*dimension + j] << ",";
		}
		outputstream << std::endl;
	}
	outputstream.close();
}

Eigen::VectorXd WarpingMapTrainingModel::getTrainingLinearDis(int index)
{
	if (index >= trainingLinearDis.size())
	{
		std::cout << "out of range" << std::endl;
	}

	return trainingLinearDis[index];
}

void WarpingMapTrainingModel::getTraingLinearForce(int index, VectorXd &force)
{
	
}

void WarpingMapTrainingModel::getTrainingLinearForceDirection(int index, Vector3d &force)
{
	force = forcefieldDirection[index];
}

Eigen::VectorXd WarpingMapTrainingModel::getTrainingNonLinearDis(int index)
{
	if (index >= trainingNonLinearDis.size())
	{
		std::cout << "out of range" << std::endl;
	}
	return trainingNonLinearDis[index];
}

Eigen::Vector3d WarpingMapTrainingModel::getForceFieldDirection(int index)
{
	if (index >= forcefieldDirection.size())
	{
		//std::cout << "out of range" << std::endl;
		return Vector3d(1, 0, 0);
	}
	return forcefieldDirection[index];
}

double WarpingMapTrainingModel::getPoissonPerDis(int index)
{
	if (index >= poissonPerDis.size())
	{
		std::cout << "out of range" << std::endl;
		return -1;
	}
	return poissonPerDis[index];
}

void WarpingMapTrainingModel::setTrainingNonLinearDis(VectorXd dis, int index)
{
	if (index >= trainingNonLinearDis.size())
	{
		std::cout << "out of range" << std::endl;
	}
	trainingNonLinearDis[index] = dis;
}

int WarpingMapTrainingModel::getTotalSizeOfTrainingDis()
{
	return trainingLinearDis.size();
}

int WarpingMapTrainingModel::getTotalSizeOfTestDis()
{
	return testLinearDis.size();
}

void WarpingMapTrainingModel::saveTrainingSet(const char* filename)
{
	std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
	out.precision(64);
	int numTrainingSet = this->getNumTrainingSet();
	int numTrainingSetHigh = this->getNumTrainingHighFreq();
	int totaltrainingSet = trainingLinearDis.size();

	out.write((char*)&numTrainingSet, sizeof(int));
	out.write((char*)&numTrainingSetHigh, sizeof(int));
	out.write((char*)&totaltrainingSet, sizeof(int));

	for (int i = 0; i < totaltrainingSet; i++)
	{
		out.write((char*)trainingLinearDis[i].data(),sizeof(double)*r);
		out.write((char*)trainingNonLinearDis[i].data(), sizeof(double)*r);
		out.write((char*)forcefieldDirection[i].data(), sizeof(double) * 3);
		out.write((char*)&poissonPerDis[i], sizeof(double));
	}

	out.close();
}

void WarpingMapTrainingModel::readTrainingSet(const char* filename)
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	in.precision(64);
	int numTrainingSet;
	int numTrainingSetHigh;
	int totaltrainingSet;

	in.read((char*)&numTrainingSet, sizeof(int));
	in.read((char*)&numTrainingSetHigh, sizeof(int));
	in.read((char*)&totaltrainingSet, sizeof(int));

	setNumTrainingSet(numTrainingSet);
	setNumTrainingHighFreq(numTrainingSetHigh);
	
	trainingLinearDis.clear();
	trainingNonLinearDis.clear();
	forcefieldDirection.clear();
	poissonPerDis.clear();

	VectorXd temp(r);
	Vector3d direction_;
	for (int i = 0; i < totaltrainingSet; i++)
	{
		in.read((char*)temp.data(), sizeof(double)*r);
		trainingLinearDis.push_back(temp);
		in.read((char*)temp.data(), sizeof(double)*r);
		trainingNonLinearDis.push_back(temp);
		in.read((char*)direction_.data(), sizeof(double) * 3);
		forcefieldDirection.push_back(direction_);
		double poisson; 
		in.read((char*)&poisson, sizeof(double));
		poissonPerDis.push_back(poisson);
	}

	in.close();
	alreadyReadData = true;
}

VectorXd WarpingMapTrainingModel::rotationGravity(VectorXd gravity, double angle, Vector3d axis)
{
	AngleAxis<double> aa(angle, axis);
	Matrix3d rmatrix_ = aa.toRotationMatrix();
	Vector3d exam(0, -1, 0);
	std::cout << rmatrix_*exam << std::endl;

	VectorXd output = gravity;
	for (int i = 0; i < gravity.rows() / 3; i++)
	{
		Vector3d nodegravity;
		for (int j = 0; j < 3; j++)
		{
			nodegravity.data()[j] = gravity.data()[i * 3 + j];
		}
		nodegravity = rmatrix_*nodegravity;
		for (int j = 0; j < 3; j++)
		{
			output.data()[i * 3 + j] = nodegravity.data()[j];
		}
	}
	return output;
}

Eigen::VectorXd WarpingMapTrainingModel::createGravity(Vector3d nodeacc)
{
	VectorXd gravity(r);
	gravity.setZero();

	for (int i = 0; i < r / 3; i++)
	{
		gravity.data()[i * 3 + 0] = nodeacc.data()[0];
		gravity.data()[i * 3 + 1] = nodeacc.data()[1];
		gravity.data()[i * 3 + 2] = nodeacc.data()[2];
	}
	gravity = *massMatrix*gravity;
	return gravity;
}
