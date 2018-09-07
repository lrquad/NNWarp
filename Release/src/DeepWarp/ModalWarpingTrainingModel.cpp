#include "ModalWarpingTrainingModel.h"
#include "Integrator/ImplicitNewMatkSparseIntegrator.h"
#include <random>
#include <fstream>
#include <iostream>
#include <time.h> 
#include "Simulator/DeepWarp/ModalRotationMatrix.h"
#include "Integrator/ImplicitModalWarpingIntegrator.h"
#include "Functions/GeoMatrix.h"
#include <Eigen/QR>
#include "Functions/findElementInVector.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/ForceField/RotateForceField.h"

#include "Simulator/ReducedSTVK/ReducedSTVKModel.h"
#include "Simulator/ReducedForceModel/ReducedForceModel.h"
#include "Simulator/ReducedForceModel/ReducedSTVKForceModel.h"
#include "Integrator/ImpicitNewMarkDenseIntegrator.h"


ModalWarpingTrainingModel::ModalWarpingTrainingModel(LoboVolumetricMesh* volumtrciMesh_, VectorXd gravity_, SparseMatrix<double>* modalRotationSparseMatrix_, ModalRotationMatrix* modalrotationMatrix_, MatrixXd* subspaceModes_, int r, double timestep, SparseMatrix<double>* massMatrix_, LoboForceModel* forcemodel_, LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef, bool useStaticSolver) :WarpingMapTrainingModel(subspaceModes_, r, timestep, massMatrix_, forcemodel_, nonlinearforceModel_, numConstrainedDOFs_, constrainedDOFs_, dampingMassCoef, dampingStiffnessCoef, useStaticSolver)
{
	this->volumtrciMesh = volumtrciMesh_;
	this->modalRotationSparseMatrix = modalRotationSparseMatrix_;
	this->modalrotationMatrix = modalrotationMatrix_;
	localOrientationMatrixR = new SparseMatrix<double>();
	matrixR_ = new SparseMatrix<double>();

	this->gravity = gravity_;

	//0.001
	modalwarpingintegrator = new ImplicitModalWarpingIntegrator(modalrotationMatrix, modalRotationSparseMatrix,
		r, 0.001, massMatrix_, loboforceModel, 1, numConstrainedDOFs_, constrainedDOFs_, dampingMassCoef, dampingStiffnessCoef, 1, 1e-7, 0.25, 0.5, false
		);


	if (subspaceModes->size() > 0)
	{
		std::cout << "read modes please" << std::endl;
		reducedMassMatrix = new MatrixXd();
		*reducedMassMatrix = subspaceModes->transpose()*(*massMatrix)**subspaceModes;
		int r = subspaceModes->cols();
		VectorXd reducedgraivty(r);
		int R = volumtrciMesh->getNumVertices() * 3;

		reducedSTVKmodel = new ReducedSTVKModel(this->volumtrciMesh, this->massMatrix, subspaceModes);

		reducedSTVKmodel->computeGravity(&reducedgraivty, massMatrix, R);
		reducedSTVKmodel->setGravityForce(reducedgraivty.data());
		reducedSTVKmodel->setGravity(false);
		reducedSTVKmodel->computeReducedModesCoefficients();

		reducedforcemodel = new ReducedSTVKForceModel(reducedSTVKmodel);

		reducedIntergrator = new ImpicitNewMarkDenseIntegrator(r, timestep, reducedMassMatrix, reducedforcemodel, 0, 0, dampingMassCoef, dampingStiffnessCoef, 1);
	}


	modalwarpingintegrator->setDampingMassCoef(0.1);
	modalwarpingintegrator->setDampingStiffnessCoef(0.1); //0.1

	nonLinearIntegrator->setDampingMassCoef(0.3);
	nonLinearIntegrator->setDampingStiffnessCoef(0.3);
	nonLinearIntegrator->setTimeStep(0.005);

	integrator->setDampingMassCoef(0.8);
	integrator->setDampingStiffnessCoef(0.8);

	nonLinearIntegrator->setMaxInteration(50);
}

ModalWarpingTrainingModel::~ModalWarpingTrainingModel()
{
	delete localOrientationMatrixR;
	delete matrixR_;
	delete reducedIntergrator;
	delete modalwarpingintegrator;
	delete reducedforcemodel;
	delete reducedSTVKmodel;
}

void ModalWarpingTrainingModel::excute()
{
	if (getForcefieldType() == 1)
	{
		modalwarpingintegrator->setTimeStep(0.002);
		modalwarpingintegrator->setDampingMassCoef(0.2);
		modalwarpingintegrator->setDampingStiffnessCoef(0.2); //0.1
		method12Twist();
	}
	else
	if (getForcefieldType() == 0)
	{
		modalwarpingintegrator->setTimeStep(0.002);
		modalwarpingintegrator->setDampingMassCoef(0.5);
		modalwarpingintegrator->setDampingStiffnessCoef(0.5); //0.1
		method10();
	}
	//methodFortwist();
	//method9();
	return;
}

void ModalWarpingTrainingModel::getTraingLinearForce(int index, VectorXd &force)
{
	//force = modalwarpingintegrator->getInteranlForce(trainingNonLinearDis[index]);

	//force = gravity;
	Vector3d nodeacc = forcefieldDirection[index];
	VectorXd fullforce = this->createGravity(nodeacc);
	force = fullforce;
	//force = gravity;
}

void ModalWarpingTrainingModel::getTraingLinearForce(VectorXd& lq, VectorXd &force)
{
	force = modalwarpingintegrator->getInteranlForce(lq);
	//force = gravity;
}

void ModalWarpingTrainingModel::getTraingLinearForce(VectorXd& lq, VectorXd &force, double poisson)
{
	LoboVolumetricMesh::Material* materia = volumtrciMesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	enmateria->setNu(poisson);
	modalwarpingintegrator->updateMaterial();
	force = modalwarpingintegrator->getInteranlForce(lq);

}

Eigen::VectorXd ModalWarpingTrainingModel::getNonlinearInternalforce(int disid)
{
	return nonLinearIntegrator->getInteranlForce(trainingNonLinearDis[disid]);
}

void ModalWarpingTrainingModel::subexcute3()
{
	std::cout << "num low fre data." << getNumTrainingSet() << std::endl;
	std::cout << "num high fre data." << getNumTrainingHighFreq() << std::endl;

	trainingLinearDis.clear();
	trainingNonLinearDis.clear();
	int maxIteration = 1000;
	trainingLinearDis.reserve(getNumTrainingSet() + getNumTrainingHighFreq());
	trainingNonLinearDis.reserve(getNumTrainingSet() + getNumTrainingHighFreq());
	VectorXd reducedForce(subspaceModes->cols());
	int dataCount = 0;

	while (dataCount < getNumTrainingSet())
	{
		std::cout << dataCount << std::endl;
		srand(time(NULL) + dataCount);
		std::default_random_engine generator;
		generator.seed(time(NULL) + dataCount);
		std::uniform_real_distribution<double> latitude_distribution(-forceScale, forceScale);
		double scale = (this->forceScale / getNumTrainingSet()) *(getNumTrainingSet() - dataCount) + 0.01;
		std::cout << "scale " << scale << std::endl;
		double angle = (PI_/2 / getNumTrainingSet())*dataCount + 0.01;

		Vector3d axis(0, 0, 1);

		angle *= -1;

		//axis.setRandom();
		//axis.normalize();
		//test

		VectorXd fullforce = this->rotationGravity(gravity, angle, axis);

		//reducedForce.setRandom();
		//fullforce = (*subspaceModes)*reducedForce;

		//fullforce.setRandom();
		//fullforce *= scale;

		//fullforce = gravity*scale;

		nonLinearIntegrator->setExternalForces(fullforce.data());
		nonLinearIntegrator->resetToRest();
		nonLinearIntegrator->setSaveStepResidual(true);
		VectorXd q_dis;

		//store training linear dis
		//trainingLinearDis.push_back(q_dis);
		double preNorm = DBL_MAX;
		VectorXd preDis = gravity;
		preDis.setConstant(1000);

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
					modalwarpingintegrator->setSaveStepResidual(true);
					convergeIntegrationLocal(modalwarpingintegrator, curExternalForce, 1000);
					q_dis = modalwarpingintegrator->getVectorq();

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

			if (dataCount > getNumTrainingSet())
			{
				break;
			}
		}

		nonLinearIntegrator->setSaveStepResidual(false);
	}
}

void ModalWarpingTrainingModel::subexcute()
{
	std::cout << "num low fre data." << getNumTrainingSet() << std::endl;
	std::cout << "num high fre data." << getNumTrainingHighFreq() << std::endl;

	trainingLinearDis.clear();
	trainingNonLinearDis.clear();
	int maxIteration = 1000;
	trainingLinearDis.reserve(getNumTrainingSet() + getNumTrainingHighFreq());
	trainingNonLinearDis.reserve(getNumTrainingSet() + getNumTrainingHighFreq());
	VectorXd reducedForce(subspaceModes->cols());
	int dataCount = 0;
	while (dataCount < getNumTrainingSet())
	{
		std::cout << dataCount << std::endl;
		srand(time(NULL) + dataCount);
		std::default_random_engine generator;
		generator.seed(time(NULL) + dataCount);
		std::uniform_real_distribution<double> latitude_distribution(-forceScale, forceScale);
		double scale = (this->forceScale / getNumTrainingSet()) *(getNumTrainingSet()-dataCount) + 0.01;
		std::cout <<"scale "<< scale << std::endl;
		double angle = (PI_ / getNumTrainingSet())*dataCount - PI_ / 2 + 0.1;

		Vector3d axis(0, 0, 1);
		//axis.setRandom();
		//axis.normalize();
		//test
		angle = PI_ / 2;

		VectorXd fullforce = this->rotationGravity(gravity, angle, axis);
		
		//reducedForce.setRandom();
		//fullforce = (*subspaceModes)*reducedForce;

		//fullforce.setRandom();
		//fullforce *= scale;

		fullforce = gravity*scale;

		modalwarpingintegrator->resetToRest();
		modalwarpingintegrator->setSaveStepResidual(true);
		double preNorm = DBL_MAX;
		double residual = DBL_MAX;
		VectorXd preq = fullforce;
		preq.setConstant(10000);

		for (int j = 0; j < maxIteration; j++)
		{
			VectorXd fullq = modalwarpingintegrator->getVectorq();
			VectorXd w = (*modalRotationSparseMatrix)*fullq;

			//set the external force
			modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w);
			//VectorXd mergedExternalForce = (*localOrientationMatrixR)*fullforce;
			modalwarpingintegrator->setExternalForces(fullforce.data());
			modalwarpingintegrator->doTimeStep();

			double norm_ = modalwarpingintegrator->getVectorq().norm();

			residual = std::abs((fullq - preq).norm() / preq.norm());

			if (residual > 0.01)
			{
				preNorm = norm_;
				preq = fullq;

				if (norm_ > 0.01)
				{
					bool reset = false;
					if (j == 0)
					{
						reset = true;
					}

					VectorXd q_dis;
					q_dis = modalwarpingintegrator->getVectorq();
					w = (*modalRotationSparseMatrix)*q_dis;
					//store training linear dis
					trainingLinearDis.push_back(q_dis);
					VectorXd curExternalForce = modalwarpingintegrator->getStep_residual();
					modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w,false);
					curExternalForce = (*localOrientationMatrixR)*curExternalForce;

					convergeIntegrationNonLinear(nonLinearIntegrator, curExternalForce, maxIteration, reset);
					q_dis = nonLinearIntegrator->getVectorq();
					trainingNonLinearDis.push_back(q_dis);
					dataCount++;
					std::cout << dataCount << std::endl;
				}
			}
			

			if (residual < 1e-6)
			{
				break;
			}

			if (dataCount > getNumTrainingSet())
			{
				break;
			}
		}

		modalwarpingintegrator->setSaveStepResidual(false);
	}
	std::cout << "excute finished" << std::endl;
}

void ModalWarpingTrainingModel::subexcute2()
{
	std::vector<Vector3d> nodedisplacement;

	generateRotationVectorSample(M_PI * 2, 0, 3,
		M_PI / 2, -M_PI / 2, 1,
		0, M_PI/3, 2, nodedisplacement);

	/*Vector3d firstNode = nodedisplacement[10];
	nodedisplacement.clear();
	nodedisplacement.push_back(firstNode);*/



	int r = this->gravity.rows();

	double maxNodedisdance = -DBL_MAX;
	int maxNodeid = -1;
	
	for (int j = 0; j < r / 3; j++)
	{
		if (node_distance[j] > maxNodedisdance)
		{
			maxNodedisdance = node_distance[j];
			maxNodeid = j;
		}
	}

	LoboNodeBase* nodep = volumtrciMesh->getNodeRef(maxNodeid);
	int correspondingSize = 3 + nodep->neighbor.size() * 3;
	int Phi_1_Size = nodep->neighbor.size() * 3;
	MatrixXd phi1(3, Phi_1_Size);
	phi1.setZero();
	MatrixXd phi2(3, 3);
	phi2.setZero();
	MatrixXd phitotal(3, Phi_1_Size + 3);
	
	for (int k = 0; k < modalRotationSparseMatrix->outerSize(); ++k)
		for (SparseMatrix<double>::InnerIterator it(*modalRotationSparseMatrix, k); it; ++it)
		{
			if (it.row() / 3 == maxNodeid)
			{
				int neighborid = it.col() / 3;
				int insideIndex = findElementIndex(nodep->neighbor, neighborid);
				if (insideIndex != -1)
				{
					phi1.data()[(insideIndex * 3 + it.col() % 3) * 3 + it.row() % 3] = it.value();
					phitotal.data()[(insideIndex * 3 + it.col() % 3) * 3 + it.row() % 3] = it.value();

				}
				if (it.col() / 3 == maxNodeid)
				{
					phi2.data()[(it.col() % 3) * 3 + it.row() % 3] = it.value();
					phitotal.data()[(nodep->neighbor.size() * 3 + it.col() % 3) * 3 + it.row() % 3] = it.value();
				}
			}
		}

	MatrixXd invPhi1 = pseudoinverse(phi1);
	MatrixXd invPhitotal = pseudoinverse(phitotal);

	if (subspaceModes == NULL)
	{
		std::cout << "subspaceModes is NULL" << std::endl;
	}

	MatrixXd subModes(Phi_1_Size + 3, subspaceModes->cols());
	subModes.setZero();
	for (int i = 0; i < nodep->neighbor.size() * 3; i++)
	{
		int neighborid = nodep->neighbor[i / 3];
		int row = neighborid * 3 + i % 3;
		for (int j = 0; j < subspaceModes->cols(); j++)
		{
			subModes.data()[j*subModes.rows() + i] = subspaceModes->data()[j*subspaceModes->rows() + row];
		}
	}

	for (int i = 0; i < 3; i++)
	{
		int row = maxNodeid * 3 + i;
		for (int j = 0; j < subspaceModes->cols(); j++)
		{
			subModes.data()[j*subModes.rows() + i + Phi_1_Size] = subspaceModes->data()[j*subspaceModes->rows() + row];
		}
	}

	MatrixXd phiSubModes = phitotal*subModes;
	MatrixXd invPhiSubModes = pseudoinverse(phiSubModes);
	MatrixXd invSubModes = pseudoinverse(subModes);
	
	modalwarpingintegrator->setStoreLagrangeMultipliers(true);

	int numscale = 1;

	std::vector<VectorXd> exforceList;

	for (int i = 0; i < subspaceModes->cols(); i++)
	{
		VectorXd lq = subspaceModes->col(i);
		VectorXd w = (*modalRotationSparseMatrix)*lq;
		modalrotationMatrix->computeWarpingRotationMatrixR(localOrientationMatrixR, w);
		VectorXd nq = *localOrientationMatrixR*lq;
		VectorXd constrainForce = nonLinearIntegrator->getInteranlForce(nq);
		exforceList.push_back(constrainForce);
	}

	if (0)
	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		VectorXd w(r);
		w.setZero();

		
		Vector3d w_j = nodedisplacement[i];

		for (int j = 0; j < numscale; j++)
		{
			Vector3d uscale = Vector3d(0, -j*0.06, 0);

			VectorXd targetq;

			VectorXd T(nodep->neighbor.size() * 3 + 3);
			T.setZero();

			for (int k = 0; k < T.rows() / 3; k++)
			{
				T.data()[k * 3 + 0] = uscale.data()[0];
				T.data()[k * 3 + 1] = uscale.data()[1];
				T.data()[k * 3 + 2] = uscale.data()[2];
			}

			targetq = invPhitotal*(w_j - phitotal*T);
			targetq += T;

			VectorXd linearq;
			VectorXd reducedq = invPhiSubModes*(w_j);
			linearq = subModes*reducedq;
			targetq = linearq + T;

			SparseMatrix<double> constrainMatrix;
			VectorXd constrainTarget;

			VectorXd lq(r);
			lq.setZero();
			lq = *subspaceModes*invSubModes*targetq;
			w = (*modalRotationSparseMatrix)*lq;
			modalrotationMatrix->computeWarpingRotationMatrixR(localOrientationMatrixR, w);
			VectorXd nq = *localOrientationMatrixR*lq;

			VectorXd constrainForce = nonLinearIntegrator->getInteranlForce(nq);
			/*constrainForce = modalwarpingintegrator->getInteranlForce(lq);
			modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
			constrainForce = *localOrientationMatrixR*constrainForce;*/
			exforceList.push_back(constrainForce);

			//VectorXd nq;
			////nq = *localOrientationMatrixR*lq;
			//nq = nonLinearIntegrator->getVectorq();

			//trainingLinearDis.push_back(lq);

			//trainingNonLinearDis.push_back(nq);
		}
	}

	int maxIteration = 1000;
	int dataCount = 0;
	/*VectorXd temp = exforceList[0];
	exforceList.clear();
	exforceList.push_back(temp);*/
	
	this->setNumTrainingSet(exforceList.size());
	this->setNumTrainingHighFreq(0);

	std::cout << "num low fre data." << getNumTrainingSet() << std::endl;
	std::cout << "num high fre data." << getNumTrainingHighFreq() << std::endl;

	for (int i = 0; i < exforceList.size(); i++)
	{
		std::cout <<"--------"<< i <<"----------"<< std::endl;
		VectorXd fullforce = exforceList[i];

		//reducedForce.setRandom();
		//fullforce = (*subspaceModes)*reducedForce;

		//fullforce.setRandom();
		//fullforce *= scale;

		nonLinearIntegrator->setExternalForces(fullforce.data());
		nonLinearIntegrator->resetToRest();
		nonLinearIntegrator->setSaveStepResidual(true);
		VectorXd q_dis;

		//store training linear dis
		//trainingLinearDis.push_back(q_dis);
		double preNorm = DBL_MAX;
		VectorXd preDis = gravity;
		preDis.setConstant(1000);

		double residual;
		VectorXd qn = subspaceModes->col(i);
		bool converged = convergeIntegrationLocal(modalwarpingintegrator, fullforce, 1000, true);
		VectorXd ql = modalwarpingintegrator->getVectorq();
		trainingNonLinearDis.push_back(qn);
		trainingLinearDis.push_back(ql);

		if (0)
		for (int j = 0; j < maxIteration; j++)
		{
			nonLinearIntegrator->doTimeStep();

			double norm_ = nonLinearIntegrator->getVectorq().norm();
			VectorXd cur_dis = nonLinearIntegrator->getVectorq();

			residual = std::abs((preNorm - norm_) / preNorm);
			//We only pick the displacement which has larget difference
			if (residual < 1e-5)
			{
				if (norm_ > 0.01)
				{
					VectorXd curExternalForce = nonLinearIntegrator->getStep_residual();
					modalrotationMatrix->computeLocalOrientationByPolarDecomposition(localOrientationMatrixR, cur_dis, true);
					curExternalForce = (*localOrientationMatrixR)*curExternalForce;

					bool ifreset = false;
					if (j == 0)
					{
						ifreset = true;
					}
					bool converged = convergeIntegrationLocal(modalwarpingintegrator, curExternalForce, 1000, ifreset);

					if (!converged)
					{
						continue;
					}

					q_dis = modalwarpingintegrator->getVectorq();
					trainingNonLinearDis.push_back(cur_dis);
					trainingLinearDis.push_back(q_dis);

					dataCount++;
				}
				break;
			}

			preNorm = norm_;

			if (dataCount > getNumTrainingSet())
			{
				break;
			}
		}

		nonLinearIntegrator->setSaveStepResidual(false);
	}

	std::cout << "excute finished" << std::endl;

}

void ModalWarpingTrainingModel::subexcute4()
{
	int numScale = 100;
	this->setNumTrainingSet(numScale);
	this->setNumTrainingHighFreq(0);

	MatrixXd forceMatrix(subspaceModes->rows(), numScale);

	for (int i = 0; i < numScale; i++)
	{
		VectorXd lq = subspaceModes->col(2) * (i*0.1+0.1);
		
		VectorXd w = *modalRotationSparseMatrix*lq;

		VectorXd extforce = modalwarpingintegrator->getInteranlForce(lq);

		modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);

		extforce = *localOrientationMatrixR*extforce;

		
		modalrotationMatrix->computeWarpingRotationMatrixR(localOrientationMatrixR, w);

		VectorXd guessq = *localOrientationMatrixR*lq;
		VectorXd modalwarpingforce = nonLinearIntegrator->getInteranlForce(guessq);
		std::cout <<"diff force = > "<< (modalwarpingforce - extforce).norm() / extforce.norm() << std::endl;
		forceMatrix.col(i) = extforce;


		//convergeIntegrationNonLinear(nonLinearIntegrator, extforce, 1000, false);

		//VectorXd nq = nonLinearIntegrator->getVectorq();

		trainingNonLinearDis.push_back(guessq);
		trainingLinearDis.push_back(lq);

	}

}

void ModalWarpingTrainingModel::subexcute5()
{
	int numScale = 600;
	int numData = 400;
	this->setNumTrainingSet(numData-1);
	this->setNumTrainingHighFreq(0);

	for (int i = 0; i < numScale; i++)
	{
		std::cout << "Iteration => " << i << std::endl;
		VectorXd lq = ((subspaceModes->col(3)*10)/numScale)*(i+1);
		VectorXd lf = modalwarpingintegrator->getInteranlForce(lq);
		VectorXd w = *modalRotationSparseMatrix*lq;
		modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
		VectorXd nf = *localOrientationMatrixR*lf;
		modalrotationMatrix->computeWarpingRotationMatrixR(matrixR_, w);
		VectorXd wq = *matrixR_*lq;

		
		if (i / numScale < 0.1)
		{
			nonLinearIntegrator->setState(wq.data());
		}

		nonLinearIntegrator->setMaxInteration(80);
		convergeIntegrationNonLinear(nonLinearIntegrator, nf, 100, false);

		VectorXd nq = nonLinearIntegrator->getVectorq();

		if (numScale - i <= numData)
		{
			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(lq);
		}
	}



}

void ModalWarpingTrainingModel::method6()
{

	int numScale = 100;
	int numData = 100;
	this->setNumTrainingSet(numData*2);
	this->setNumTrainingHighFreq(0);

	int numVertex = volumtrciMesh->getNumVertices();
	std::ofstream test("test2.txt");
	VectorXd nodeYaxis(numVertex * 3);
	nodeYaxis.setZero();
	for (int i = 0; i < numVertex; i++)
	{
		nodeYaxis.data()[i * 3 + 1] = 1;
	}

	VectorXd lq = (subspaceModes->col(3) * 10);

	VectorXd lf = modalwarpingintegrator->getInteranlForce(lq);
	VectorXd w = *modalRotationSparseMatrix*lq;
	modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	VectorXd lqorientation = *localOrientationMatrixR*nodeYaxis;

	VectorXd nf = *localOrientationMatrixR*lf;
	modalrotationMatrix->computeWarpingRotationMatrixR(matrixR_, w);
	VectorXd wq = *matrixR_*lq;
	modalrotationMatrix->computeLocalOrientationByPolarDecomposition(localOrientationMatrixR, wq, false);
	VectorXd nqorientation = *localOrientationMatrixR*nodeYaxis;
	test << lqorientation.transpose() << std::endl;
	test << nqorientation.transpose() << std::endl;
	test.close();
	std::cout << (lqorientation - nqorientation).norm() / nqorientation.norm() << std::endl;

	VectorXd ori_p(numVertex * 3);
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d nodeori = volumtrciMesh->getNodeRestPosition(i);
		ori_p.data()[i * 3 + 0] = nodeori.data()[0];
		ori_p.data()[i * 3 + 1] = nodeori.data()[1];
		ori_p.data()[i * 3 + 2] = nodeori.data()[2];
	}

	for (int i = 0; i < numScale; i++)
	{
		std::cout << "Iteration => " << i << std::endl;
		VectorXd nq = ((*matrixR_*lq) / numScale)*(i + 1);
		VectorXd originlq = lq / numScale*(i + 1);
		
		VectorXd real_nf = nonLinearIntegrator->getInteranlForce(nq);
		modalrotationMatrix->computeLocalOrientationByPolarDecomposition(localOrientationMatrixR, nq, true);
		//modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, true);
		VectorXd warp_nf = *localOrientationMatrixR*real_nf;

		convergeIntegrationLocal(modalwarpingintegrator, warp_nf, 200, true);

		VectorXd real_lq = modalwarpingintegrator->getVectorq();
		//VectorXd real_lq;
		//real_lq = *localOrientationMatrixR*nq;

		if (numScale - i <= numData)
		{
			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(real_lq);
			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(originlq);
		}
	}
}

void ModalWarpingTrainingModel::method7()
{
	int numScale = 1;
	int numData = 1;
	this->setNumTrainingSet(numData * 2);
	this->setNumTrainingHighFreq(0);

	Vector3d axis(0, 0, 1);
	double angle = PI_/2;
	VectorXd fullforce = this->rotationGravity(gravity, angle, axis);

	convergeIntegrationNonLinear(nonLinearIntegrator, fullforce, 1000, true);
	VectorXd targetnq = nonLinearIntegrator->getVectorq();

	for (int i = numScale-1; i < numScale; i++)
	{
		std::cout << "Iteration => " << i << std::endl;
		VectorXd nq = ((targetnq) / numScale)*(i + 1);
		VectorXd real_nf = nonLinearIntegrator->getInteranlForce(nq);
		modalrotationMatrix->computeLocalOrientationByPolarDecomposition(localOrientationMatrixR, nq, true);
		VectorXd warp_nf = *localOrientationMatrixR*real_nf;
		convergeIntegrationLocal(modalwarpingintegrator, warp_nf, 200, true);
		VectorXd real_lq = modalwarpingintegrator->getVectorq();

		//warp real_lq
		VectorXd w = *modalRotationSparseMatrix*real_lq;
		modalrotationMatrix->computeWarpingRotationMatrixR(matrixR_, w);
		VectorXd wq = *matrixR_*real_lq;

		if (numScale - i <= numData)
		{
			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(real_lq);
			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(wq);
		}
	}

}

void ModalWarpingTrainingModel::method8()
{
	std::vector<Vector3d> nodedisplacement;
	nodedisplacement.clear();
	std::vector<Vector3d> potentialDirection;
	std::vector<double> poissonPerdirection;
	std::vector<bool> ifreset;

	//if (0)
	{
		generateRotationVectorSample(M_PI, M_PI, 1,
			M_PI/3.0, -M_PI / 2.0, 10,
			9.8, 4.9, 2, nodedisplacement, ifreset, potentialDirection);
		samplePoissonRatio(0.3, 0.3, 1, 1, nodedisplacement, potentialDirection, ifreset, poissonPerdirection);
	}

	/*nodedisplacement.push_back(Vector3d(0, -1, 0)*9.8);
	ifreset.push_back(true);
	potentialDirection.push_back(Vector3d(0, -1, 0));
	poissonPerdirection.push_back(0.20);*/

	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		std::cout << nodedisplacement[i].transpose() << std::endl;
		std::cout << "poisson " << poissonPerdirection[i] << std::endl;
	}
	
	int numDataPerSample = getNumTrainingSet() / nodedisplacement.size();
	setNumTrainingHighFreq(0);

	std::vector<VectorXd> qn_list;
	std::vector<VectorXd> ql_list;

	int dataCount = 0;
	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		VectorXd nodeacc = nodedisplacement[i];
		VectorXd fullforce = this->createGravity(nodeacc);


		LoboVolumetricMesh::Material* materia = volumtrciMesh->getMaterialById(0);
		LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
		enmateria->setNu(poissonPerdirection[i]);
		modalwarpingintegrator->updateMaterial();
		nonLinearIntegrator->updateMaterial();

		convergeIntegrationNonLinearBuffer(nonLinearIntegrator, fullforce, 2000, qn_list, true);

		int range = qn_list.size() / numDataPerSample;
		range = 1;
		for (int j = 0; j <qn_list.size(); j++)
		{
			VectorXd nq = qn_list[j];

			if (trainingLinearDis.size() > 0)
				if ((nq - trainingNonLinearDis.back()).norm() / trainingNonLinearDis.back().norm() < 1e-3)
				{
					//the diff is too small
					continue;
				}

			VectorXd real_nf = nonLinearIntegrator->getInteranlForce(nq);
			modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, nq, true);
			VectorXd warp_nf = *localOrientationMatrixR*real_nf;
			
			bool reset = false;

			if (j == 0 && ifreset[i] == true)
			{
				reset = true;
			}

			convergeIntegrationLocal(modalwarpingintegrator, real_nf, 2000, reset);
			
			VectorXd real_lq = modalwarpingintegrator->getVectorq();

			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(real_lq);
			forcefieldDirection.push_back(potentialDirection[i]);
			poissonPerDis.push_back(poissonPerdirection[i]);

			std::cout << dataCount << "/" << getNumTrainingSet() << std::endl;
			dataCount++;
		}
	}
	setNumTrainingSet(trainingNonLinearDis.size());

	std::cout << "finished" << std::endl;
}

void ModalWarpingTrainingModel::method9()
{
	std::vector<Vector3d> nodedisplacement;
	std::vector<Vector3d> potentialDirection;
	std::vector<double> poissonPerdirection;
	nodedisplacement.clear();
	std::vector<bool> ifreset;

	nodedisplacement.clear();
	ifreset.clear();

	//if (0)
	{
		generateRotationVectorSample(M_PI, M_PI, 1,
			M_PI / 2.0, -M_PI / 2.0, 10,
			4.0, 2.0, 2, nodedisplacement, ifreset, potentialDirection);

		samplePoissonRatio(0.2, 0.2, 1, 1, nodedisplacement, potentialDirection, ifreset, poissonPerdirection);
	}

	/*nodedisplacement.push_back(Vector3d(0, -1, 0)*9.8);
	ifreset.push_back(true);
	potentialDirection.push_back(Vector3d(0, -1, 0));
	poissonPerdirection.push_back(0.20);*/

	//nodedisplacement.push_back(Vector3d(-4.9, 1.83721e-032, 3.00038e-016));
	//ifreset.push_back(true);
	//potentialDirection.push_back(Vector3d(-1, 0, 0));
	//poissonPerdirection.push_back(0.2);


	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		std::cout << nodedisplacement[i].transpose() << std::endl;
		std::cout <<"poisson " << poissonPerdirection[i] << std::endl;
	}

	int numDataPerSample = getNumTrainingSet() / nodedisplacement.size();
	setNumTrainingHighFreq(0);

	std::vector<VectorXd> qn_list;
	int dataCount = 0;
	nonLinearIntegrator->resetToRest();
	//VectorXd q_test(volumtrciMesh->getNumVertices() * 3);
	//q_test.setRandom();
	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		Vector3d nodeacc = nodedisplacement[i];
		VectorXd fullforce = this->createGravity(nodeacc);
		
		LoboVolumetricMesh::Material* materia = volumtrciMesh->getMaterialById(0);
		LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
		enmateria->setNu(poissonPerdirection[i]);
		modalwarpingintegrator->updateMaterial();
		nonLinearIntegrator->updateMaterial();

		//reset = false;
		bool reset = ifreset[i];

		std::cout <<"ifrest"<< reset << std::endl;
		convergeIntegrationLocalBuffer(modalwarpingintegrator, fullforce, 1000, qn_list, reset);

		int range = qn_list.size() / numDataPerSample;
		if (range <= 1)
		{
			range = 2;
		}
		range = 1;

		std::cout <<"range => "<< range << std::endl;
		for (int j = 0; j < qn_list.size(); j += range)
		{
			std::cout << "force " << i << "/" << nodedisplacement.size() - 1 << std::endl;
			std::cout << "q_list " << j << "/" << qn_list.size() - 1 << std::endl;
			VectorXd lq = qn_list[j];

			if (trainingLinearDis.size() > 0)
				if ((lq - trainingLinearDis.back()).norm() / trainingLinearDis.back().norm() < 1e-3)
				{
					//the diff is too small
					continue;
				}

			//use ku = f instead of ku = Rf
			VectorXd nq;
			if (getLinearDisOnly())
			{
				VectorXd w = *modalRotationSparseMatrix*lq;
				modalrotationMatrix->computeWarpingRotationMatrixR(localOrientationMatrixR, w);
				nq = *localOrientationMatrixR*lq;
			}
			else
			{
				VectorXd w = *modalRotationSparseMatrix*lq;
				modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
				VectorXd lf = modalwarpingintegrator->getInteranlForce(lq);
				VectorXd nf = *localOrientationMatrixR*lf;

				bool reset = false;

				if (j == 0 && ifreset[i] == true)
				{
					reset = true;
				}

				bool converged = this->convergeIntegrationNonLinear(nonLinearIntegrator, nf, 1000, reset);

				nq = nonLinearIntegrator->getVectorq();

				if (converged == false)
				{
					std::cout << "break" << std::endl;
					break;
				}
			}

			if (dataCount % 2 == 1)
			{
				dataCount++;
				continue;
			}

			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(lq);
			forcefieldDirection.push_back(potentialDirection[i]);
			
			poissonPerDis.push_back(poissonPerdirection[i]);

			std::cout << dataCount << "/" << getNumTrainingSet() << std::endl;
			dataCount++;
		}
	}
	setNumTrainingSet(trainingNonLinearDis.size());
}

void ModalWarpingTrainingModel::method10()
{
	std::vector<Vector3d> nodedisplacement;
	std::vector<Vector3d> potentialDirection;
	std::vector<double> poissonPerdirection;
	nodedisplacement.clear();
	std::vector<bool> ifreset;

	nodedisplacement.clear();
	ifreset.clear();

	if (0)
	{
		generateRotationVectorSample(M_PI, M_PI, 1,
			M_PI / 2.0, -M_PI / 2.0, 10,
			4.0, 2.0, 2, nodedisplacement, ifreset, potentialDirection);

		samplePoissonRatio(0.2, 0.2, 1, 1, nodedisplacement, potentialDirection, ifreset, poissonPerdirection);
	}

	nodedisplacement.push_back(Vector3d(0, -1, 0)*9.8);
	ifreset.push_back(true);
	potentialDirection.push_back(Vector3d(0, -1, 0));
	poissonPerdirection.push_back(0.20);

	//nodedisplacement.push_back(Vector3d(-4.9, 1.83721e-032, 3.00038e-016));
	//ifreset.push_back(true);
	//potentialDirection.push_back(Vector3d(-1, 0, 0));
	//poissonPerdirection.push_back(0.2);


	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		std::cout << nodedisplacement[i].transpose() << std::endl;
		std::cout << "poisson " << poissonPerdirection[i] << std::endl;
	}

	int numDataPerSample = getNumTrainingSet() / nodedisplacement.size();
	setNumTrainingHighFreq(0);

	std::vector<VectorXd> qn_list;
	int dataCount = 0;
	nonLinearIntegrator->resetToRest();
	//VectorXd q_test(volumtrciMesh->getNumVertices() * 3);
	//q_test.setRandom();
	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		Vector3d nodeacc = nodedisplacement[i];
		VectorXd fullforce = this->createGravity(nodeacc);

		LoboVolumetricMesh::Material* materia = volumtrciMesh->getMaterialById(0);
		LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
		enmateria->setNu(poissonPerdirection[i]);
		modalwarpingintegrator->updateMaterial();
		//nonLinearIntegrator->updateMaterial();

		//reset = false;
		bool reset = ifreset[i];

		std::cout << "ifrest" << reset << std::endl;
		convergeIntegrationLocalBuffer(modalwarpingintegrator, fullforce, 1000, qn_list, reset);

		int range = qn_list.size() / numDataPerSample;
		if (range <= 1)
		{
			range = 2;
		}
		range = 1;

		std::cout << "range => " << range << std::endl;
		for (int j = 0; j < qn_list.size(); j += range)
		{
			std::cout << "force " << i << "/" << nodedisplacement.size() - 1 << std::endl;
			std::cout << "q_list " << j << "/" << qn_list.size() - 1 << std::endl;
			VectorXd lq = qn_list[j];

			if (trainingLinearDis.size() > 0)
				if ((lq - trainingLinearDis.back()).norm() / trainingLinearDis.back().norm() < 1e-3)
				{
					//the diff is too small
					continue;
				}

			//use ku = f instead of ku = Rf
			VectorXd nq;
			if (getLinearDisOnly())
			{
				VectorXd w = *modalRotationSparseMatrix*lq;
				modalrotationMatrix->computeWarpingRotationMatrixR(localOrientationMatrixR, w);
				nq = *localOrientationMatrixR*lq;
			}
			else
			{
				VectorXd w = *modalRotationSparseMatrix*lq;
				modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
				VectorXd lf = modalwarpingintegrator->getInteranlForce(lq);
				VectorXd nf = *localOrientationMatrixR*lf;

				bool reset = false;

				if (j == 0 && ifreset[i] == true)
				{
					reset = true;
				}

				VectorXd reducednf = subspaceModes->transpose()*nf;

				bool converged = this->convergeIntegrationNonlinearReduced(reducedIntergrator, nf, 1000, reset);

				nq = nonLinearIntegrator->getVectorq();

				if (converged == false)
				{
					std::cout << "break" << std::endl;
					break;
				}
			}

			if (dataCount % 2 == 1)
			{
				dataCount++;
				continue;
			}

			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(lq);
			forcefieldDirection.push_back(potentialDirection[i]);

			poissonPerDis.push_back(poissonPerdirection[i]);

			std::cout << dataCount << "/" << getNumTrainingSet() << std::endl;
			dataCount++;
		}
	}
	setNumTrainingSet(trainingNonLinearDis.size());
}

void ModalWarpingTrainingModel::method10Twist()
{
	std::vector<VectorXd> qn_list;
	int dataCount = 0;

	RotateForceField* forcefield = new RotateForceField(volumtrciMesh, Vector3d(1, 0, 0));
	forcefield->setForceMagnitude(0.0005);
	
	convergeIntegrationNonLinearBuffer(nonLinearIntegrator, forcefield, 300, qn_list, true);

	delete forcefield;

	int range = 1;
	std::cout << "range => " << range << std::endl;
	for (int j = 0; j < qn_list.size(); j += range)
	{
		std::cout << "q_list " << j << "/" << qn_list.size() - 1 << std::endl;
		VectorXd nq = qn_list[j];

		if (trainingNonLinearDis.size() > 0)
			if ((nq - trainingNonLinearDis.back()).norm() / trainingNonLinearDis.back().norm() < 1e-3)
			{
				//the diff is too small
				continue;
			}

		VectorXd nf = nonLinearIntegrator->getInteranlForce(nq);	

		bool reset = false;
		if (j == 0)
			reset = true;

		convergeIntegrationLocal(modalwarpingintegrator, nf, 1000, reset);
		VectorXd lq = modalwarpingintegrator->getVectorq();

		trainingNonLinearDis.push_back(nq);
		trainingLinearDis.push_back(lq);
		std::cout << dataCount << "/" << getNumTrainingSet() << std::endl;
		dataCount++;
	}
	setNumTrainingSet(trainingNonLinearDis.size());
	setNumTrainingHighFreq(0);
}

void ModalWarpingTrainingModel::method11Twist()
{
	std::vector<VectorXd> qn_list;
	int dataCount = 0;

	RotateForceField* forcefield = new RotateForceField(volumtrciMesh, Vector3d(1, 0, 0));
	forcefield->setForceMagnitude(0.002);
	convergeIntegrationLocalBuffer(modalwarpingintegrator, forcefield, 200, qn_list, true);

	delete forcefield;

	//qn_list.clear();
	//
	//double maxscale = 8;
	//for (int i = 0; i < 4000; i++)
	//{
	//	double scale = maxscale / 4000 * i;
	//	qn_list.push_back(subspaceModes->col(2)*scale);
	//}

	int range = 1;
	std::cout << "range => " << range << std::endl;
	for (int j = 0; j < qn_list.size(); j += range)
	{
		std::cout << "q_list " << j << "/" << qn_list.size() - 1 << std::endl;
		VectorXd lq = qn_list[j];

		if (trainingLinearDis.size() > 0)
			if ((lq - trainingLinearDis.back()).norm() / trainingLinearDis.back().norm() < 1e-4)
			{
				continue;
			}

		VectorXd w = *modalRotationSparseMatrix*lq;
		modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
		VectorXd lf = modalwarpingintegrator->getInteranlForce(lq);
		
		VectorXd nf = *localOrientationMatrixR*lf;

		bool reset = false;
		if (j == 0)
			reset = true;
		reset = false;

		bool converge = this->convergeIntegrationNonLinear(nonLinearIntegrator, nf, 1000, reset);

		VectorXd nq = nonLinearIntegrator->getVectorq();

		if (converge == false)
		{
			break;
		}

		/*if (j % 2 != 0)
		{
			continue;
		}*/

		trainingNonLinearDis.push_back(nq);
		trainingLinearDis.push_back(lq);

		std::cout << dataCount << "/" << getNumTrainingSet() << std::endl;
		dataCount++;
	}
	setNumTrainingSet(trainingNonLinearDis.size());
	setNumTrainingHighFreq(0);
}

void ModalWarpingTrainingModel::method12Twist()
{
	std::vector<Vector3d> nodedisplacement;
	std::vector<Vector3d> potentialDirection;
	std::vector<double> poissonPerdirection;
	nodedisplacement.clear();
	std::vector<bool> ifreset;

	/*generateRotationVectorSample(M_PI/2 , 0, 3,
		M_PI / 2, 0, 3,
		1, 1, 1, nodedisplacement);*/

	generateRotationVectorSample(M_PI, M_PI, 1,
		-M_PI / 2, -M_PI / 2, 1,
		1, 1, 1, nodedisplacement, ifreset, potentialDirection);

	samplePoissonRatio(0.20, 0.20, 1, 1, nodedisplacement, potentialDirection, ifreset, poissonPerdirection);

	//nodedisplacement.clear();
	//nodedisplacement.push_back(Vector3d(1, 0, 0));

	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		std::cout << nodedisplacement[i].transpose() << std::endl;
		std::cout << "poisson " << poissonPerdirection[i] << std::endl;
	}

	int dataCount = 0;

	RotateForceField* forcefield = new RotateForceField(volumtrciMesh, Vector3d(1, 0, 0));
	forcefield->setForceMagnitude(0.001);
	forcefield->centeraxis_position = this->getPotentialCenter();

	std::cout << "nodedisplacement.size()" << nodedisplacement.size() << std::endl;

	for (int i = 0; i < nodedisplacement.size(); i++)
	{
		forcefield->centeraxis = nodedisplacement[i].normalized();
		std::cout << forcefield->centeraxis.transpose() << std::endl;

		LoboVolumetricMesh::Material* materia = volumtrciMesh->getMaterialById(0);
		LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
		enmateria->setNu(poissonPerdirection[i]);
		modalwarpingintegrator->updateMaterial();
		nonLinearIntegrator->updateMaterial();

		double preNorm = DBL_MAX;
		double residual = DBL_MAX;

		int r = volumtrciMesh->getNumVertices() * 3;
		int maxIteration = 200;

		VectorXd preq(r);
		VectorXd extForce(r);
		VectorXd nq(r);
		nq.setZero();
		extForce.setZero();
		preq.setConstant(1000);

		bool converged = false;
		std::cout << std::endl;

		modalwarpingintegrator->resetToRest();
		for (int j = 0; j < maxIteration; j++)
		{
			bool reset = false;
			if (j == 0)
				reset = true;

			std::cout << '\r';
			std::cout << j;

			volumtrciMesh->setDisplacement(nq.data());
			forcefield->computeCurExternalForce(extForce);

			VectorXd mergedExternalForce = extForce;
			modalwarpingintegrator->setExternalForces(mergedExternalForce.data());

			modalwarpingintegrator->doTimeStep();

			VectorXd curq = modalwarpingintegrator->getVectorq();

			residual = std::abs((preq - curq).norm() / preq.norm());
			preq = curq;

			VectorXd w = *modalRotationSparseMatrix*curq;

			double maxangle = getMaxRotatedAngleFromW(w);
			std::cout << "  maxangle => " << maxangle << "  ";

			/*if (maxangle > M_PI)
			{
				break;
			}*/

			modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
			VectorXd lf = modalwarpingintegrator->getInteranlForce(curq);
			VectorXd nf = *localOrientationMatrixR*lf;

			//bool converge = true;
			
			bool converge = this->convergeIntegrationNonLinear(nonLinearIntegrator, nf, 1000, reset);
			
			nq = nonLinearIntegrator->getVectorq();

			if (converge == false)
			{
				std::cout << "not coverged" << " " << std::endl;
				break;
			}

			/*if (j % 2 != 0)
			{
			continue;
			}*/

			trainingNonLinearDis.push_back(nq);
			trainingLinearDis.push_back(curq);
			forcefieldDirection.push_back(forcefield->centeraxis);
			poissonPerDis.push_back(poissonPerdirection[i]);


			std::cout << "         " << trainingNonLinearDis.size() << " ";

			if (residual < 1e-5)
			{
				converged = true;
				std::cout << "converged" << std::endl;
				break;
			}
		}
	}

	std::cout << std::endl;
	setNumTrainingSet(trainingNonLinearDis.size());
	setNumTrainingHighFreq(0);
}

void ModalWarpingTrainingModel::methodFortwist()
{
	VectorXd baselq = subspaceModes->col(2);
	for (int i = -50; i < 50; i++)
	{
		VectorXd lq = baselq*i*0.1;
		trainingNonLinearDis.push_back(lq);
		trainingLinearDis.push_back(lq);
	}

	setNumTrainingSet(trainingNonLinearDis.size());
	setNumTrainingHighFreq(0);

}

bool ModalWarpingTrainingModel::convergeIntegrationLocal(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset)
{
	if (reset)
	integrator_loc->resetToRest();

	/*modalwarpingintegrator->computeStaticDisplacement(extForce, false);

	return true;*/

	std::cout << "convergeIntegrationLocal === >" << std::endl;
	double preNorm = DBL_MAX;
	double residual = DBL_MAX;
	VectorXd preq(extForce.rows());
	preq.setConstant(1000);

	bool converged = false;

	for (int j = 0; j < maxIteration; j++)
	{
		VectorXd fullq = integrator_loc->getVectorq();
		//set the external force
		VectorXd mergedExternalForce = extForce;
		integrator_loc->setExternalForces(mergedExternalForce.data());

		integrator_loc->doTimeStep();

		VectorXd curq = integrator_loc->getVectorq();
		residual = std::abs((preq - curq).norm() / preq.norm());
		preq = integrator_loc->getVectorq();
		if (residual < 1e-6)
		{
			converged = true;
			std::cout << "converged" << std::endl;
			break;
		}
	}
	std::cout << residual << std::endl;
	return converged;
}

bool ModalWarpingTrainingModel::convergeIntegrationLocalBuffer(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, std::vector<VectorXd> &qn_list, bool reset /*= true*/)
{
	if (reset)
		integrator_loc->resetToRest();

	std::cout << "convergeIntegrationLocal === >" << std::endl;
	double preNorm = DBL_MAX;
	double residual = DBL_MAX;
	VectorXd preq(extForce.rows());
	preq.setConstant(1000);

	bool converged = false;
	qn_list.clear();
	std::cout << std::endl;
	for (int j = 0; j < maxIteration; j++)
	{
		std::cout << '\r';
		std::cout << j;
		VectorXd fullq = integrator_loc->getVectorq();
		//set the external force
		VectorXd mergedExternalForce = extForce;
		integrator_loc->setExternalForces(mergedExternalForce.data());

		integrator_loc->doTimeStep();

		VectorXd curq = integrator_loc->getVectorq();
		qn_list.push_back(curq);
		residual = std::abs((preq - curq).norm() / preq.norm());
		preq = integrator_loc->getVectorq();
		if (residual < 1e-5)
		{
			converged = true;
			std::cout << "converged" << std::endl;
			break;
		}
	}
	std::cout << std::endl;

	std::cout << residual << std::endl;
	return converged;
}

bool ModalWarpingTrainingModel::convergeIntegrationLocalBuffer(LoboIntegrator* integrator_loc, RotateForceField* forcefiled, int maxIteration, std::vector<VectorXd> &qn_list, bool reset /*= true*/)
{
	if (reset)
		integrator_loc->resetToRest();

	std::cout << "convergeIntegrationLocal === >" << std::endl;
	double preNorm = DBL_MAX;
	double residual = DBL_MAX;

	int r = volumtrciMesh->getNumVertices() * 3;

	VectorXd preq(r);
	VectorXd extForce(r);
	extForce.setZero();
	preq.setConstant(1000);

	bool converged = false;
	qn_list.clear();
	std::cout << std::endl;
	for (int j = 0; j < maxIteration; j++)
	{
		std::cout << '\r';
		std::cout << j;
		VectorXd fullq = integrator_loc->getVectorq();
		//set the external force
		VectorXd w = *modalRotationSparseMatrix*fullq;
		modalrotationMatrix->computeWarpingRotationMatrixR(localOrientationMatrixR, w);

		VectorXd nq = (*localOrientationMatrixR)*fullq;
		volumtrciMesh->setDisplacement(nq.data());
		forcefiled->computeCurExternalForce(extForce);

		VectorXd mergedExternalForce = extForce;
		integrator_loc->setExternalForces(mergedExternalForce.data());

		integrator_loc->doTimeStep();

		VectorXd curq = integrator_loc->getVectorq();
		qn_list.push_back(curq);
		residual = std::abs((preq - curq).norm() / preq.norm());
		preq = integrator_loc->getVectorq();

		if (residual < 1e-5)
		{
			converged = true;
			std::cout << "converged" << std::endl;
			break;
		}
	}
	std::cout << std::endl;

	std::cout << residual << std::endl;
	return converged;
}

bool ModalWarpingTrainingModel::convergeIntegrationNonLinear(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset)
{
	if (reset)
		integrator_loc->resetToRest();

	bool converge = ((ImplicitNewMatkSparseIntegrator*)integrator_loc)->computeStaticDisplacement(extForce, false);

	return converge;

	std::cout << "convergeIntegrationNonLinear === >" << std::endl;

	double preNorm = DBL_MAX;
	double residual = DBL_MAX;
	VectorXd preq(extForce.rows());
	preq.setConstant(1000);
	std::cout << std::endl;

	for (int j = 0; j < maxIteration; j++)
	{
		std::cout << '\r';
		std::cout << j;
		integrator_loc->setExternalForces(extForce.data());
		integrator_loc->doTimeStep();
		VectorXd curq = integrator_loc->getVectorq();
		residual = std::abs((preq - curq).norm() / preq.norm());
		preq = integrator_loc->getVectorq();
		if (residual < 1e-5)
		{
			std::cout << curq.norm() << std::endl;
			std::cout << "converged" << std::endl;
			break;
		}
	}
	std::cout << std::endl;

	std::cout << residual << std::endl;

	return true;
}

bool ModalWarpingTrainingModel::convergeIntegrationNonlinearReduced(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, bool reset /*= true*/)
{
	if (reset)
		integrator_loc->resetToRest();

	bool converge = ((ImpicitNewMarkDenseIntegrator*)integrator_loc)->computeStaticDisplacement(extForce, false);

	return converge;
}

void ModalWarpingTrainingModel::convergeIntegrationNonLinearBuffer(LoboIntegrator* integrator_loc, VectorXd &extForce, int maxIteration, std::vector<VectorXd> &qn_list, bool reset /*= true*/)
{
	qn_list.clear();

	if (reset)
		integrator_loc->resetToRest();

	double preNorm = DBL_MAX;
	double residual = DBL_MAX;
	VectorXd preq(extForce.rows());
	preq.setConstant(1000);
	std::cout << std::endl;
	for (int j = 0; j < maxIteration; j++)
	{
		std::cout << '\r';
		std::cout << j;
		integrator_loc->setExternalForces(extForce.data());
		integrator_loc->doTimeStep();
		VectorXd curq = integrator_loc->getVectorq();
		qn_list.push_back(curq);
		residual = std::abs((preq - curq).norm() / preq.norm());
		preq = integrator_loc->getVectorq();
		if (residual < 1e-5)
		{
			std::cout << curq.norm() << std::endl;
			std::cout << "converged" << std::endl;
			break;
		}
	}
	std::cout << std::endl;
	std::cout << residual << std::endl;
}

void ModalWarpingTrainingModel::convergeIntegrationNonLinearBuffer(LoboIntegrator* integrator_loc, RotateForceField* forcefiled, int maxIteration, std::vector<VectorXd> &qn_list, bool reset /*= true*/)
{
	qn_list.clear();

	if (reset)
		integrator_loc->resetToRest();

	double preNorm = DBL_MAX;
	double residual = DBL_MAX;
	
	int r = volumtrciMesh->getNumVertices() * 3;

	VectorXd preq(r);
	VectorXd extForce(r);

	preq.setConstant(1000);
	std::cout << std::endl;
	for (int j = 0; j < maxIteration; j++)
	{
		std::cout << '\r';
		std::cout << j;

		VectorXd iq = integrator_loc->getVectorq();
		volumtrciMesh->setDisplacement(iq.data());
		forcefiled->computeCurExternalForce(extForce);

		integrator_loc->setExternalForces(extForce.data());
		integrator_loc->doTimeStep();
		VectorXd curq = integrator_loc->getVectorq();

		qn_list.push_back(curq);
		residual = std::abs((preq - curq).norm() / preq.norm());
		preq = curq;
		if (residual < 1e-4)
		{
			std::cout << curq.norm() << std::endl;
			std::cout << "converged" << std::endl;
			break;
		}
	}
	std::cout << std::endl;
	std::cout << residual << std::endl;
}

typedef Eigen::Triplet<double> EIGEN_TRI;

void ModalWarpingTrainingModel::generateConstrainMatrixAndVector(SparseMatrix<double>* constrainMatrix, VectorXd &constrainTarget, LoboNodeBase* nodep, int nodeid, VectorXd target_)
{
	int numVertex = this->volumtrciMesh->getNumVertices();
	int totalConstrain = numConstrainedDOFs + nodep->neighbor.size() * 3 +3;

	constrainMatrix->resize(totalConstrain, numVertex * 3);
	constrainTarget.resize(totalConstrain, 1);
	constrainTarget.setZero();
	std::vector<EIGEN_TRI> entrys;

	for (int i = 0; i < numConstrainedDOFs; i++)
	{
		int row = i;
		int col = constrainedDOFs[i];
		entrys.push_back(EIGEN_TRI(row, col, 1));
		constrainTarget.data()[row] = 0;
	}

	for (int i = 0; i < nodep->neighbor.size()*3; i++)
	{
		int row = i + numConstrainedDOFs;
		int col = nodep->neighbor[i/3]*3 + i % 3;
		entrys.push_back(EIGEN_TRI(row, col, 1));
		constrainTarget.data()[row] = target_.data()[i];
	}

	for (int i = 0; i < 3; i++)
	{
		int row = i + numConstrainedDOFs + nodep->neighbor.size() * 3;
		int col = nodeid * 3 + i;
		entrys.push_back(EIGEN_TRI(row, col, 1));
		constrainTarget.data()[row] = target_.data()[nodep->neighbor.size() * 3+i];
	}
	constrainMatrix->setFromTriplets(entrys.begin(), entrys.end());

}

double ModalWarpingTrainingModel::getMaxRotatedAngleFromW(VectorXd &w)
{
	double maxAngle = -DBL_MAX;
	int numVertex = w.rows() / 3;
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d wi;
		wi.data()[0] = w.data()[i * 3 + 0];
		wi.data()[1] = w.data()[i * 3 + 1];
		wi.data()[2] = w.data()[i * 3 + 2];
		double angle = wi.norm();
		if (angle > maxAngle)
		{
			maxAngle = angle;
		}
	}
	return maxAngle;
}

void ModalWarpingTrainingModel::samplePoissonRatio(double maxPoisson, double minPoisson, int maxNumPoisson, int minNumPoisson, std::vector<Vector3d> &nodeforce, std::vector<Vector3d> &forcedirection, std::vector<bool> &ifreset, std::vector<double> &poissonPerdirection)
{
	int num_direction = nodeforce.size();
	poissonPerdirection.clear();

	Vector3d baseline(-1, 0, 0);
	double maxpoisson = maxPoisson;
	double minpoisson = minPoisson;
	int maxnumpoisson = maxNumPoisson;
	int minnumpoisson = minNumPoisson;
	this->poissonPerDis.clear();

	std::vector<Vector3d> finalnodeforce;
	std::vector<Vector3d> finalforcedirection;
	std::vector<bool> finalifreset;

	for (int i = 0; i < num_direction; i++)
	{
		double angle = forcedirection[i].dot(baseline);
		angle = std::acos(angle);
		int  numpoisson = angle / (M_PI / 2.0)*(minnumpoisson - maxnumpoisson) + maxnumpoisson;
		for (int j = 0; j < numpoisson; j++)
		{
			finalnodeforce.push_back(nodeforce[i]);
			finalforcedirection.push_back(forcedirection[i]);
			finalifreset.push_back(ifreset[i]);

			double interval = (maxpoisson - minpoisson) / (numpoisson - 1);
			if (numpoisson == 1)
			{
				interval = (maxpoisson - minpoisson);
			}
			double poisson = interval*j + minpoisson;
			poissonPerdirection.push_back(poisson);
		}
	}

	nodeforce = finalnodeforce;
	forcedirection = finalforcedirection;
	ifreset = finalifreset;
}
