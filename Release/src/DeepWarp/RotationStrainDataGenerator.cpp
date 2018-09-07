#include "RotationStrainDataGenerator.h"

#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"
#include "Functions/GeoMatrix.h"
#include <fstream>
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"
#include "Simulator/ForceField/RotateForceField.h"


RotationStrainDataGenerator::RotationStrainDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/) :RotationTrainingDataGeneratorV3(tetmeshvox_, cubemesh_, volumtricMesh_, trainingModal_, modalrotaionMatrix_, modalRotationSparseMatrix_, numConstrainedDOFs, constrainedDOFs)
{
	int numVertex = volumtricMesh->getNumVertices();
	ifnodeconstrained.resize(numVertex);
	std::fill(ifnodeconstrained.begin(), ifnodeconstrained.end(), false);
	for (int i = 0; i < numConstrainedDOFs / 3; i++)
	{
		int nodeindex = constrainedDOFs[i * 3] / 3;
		ifnodeconstrained[nodeindex] = true;
	}

	applyalignRotation = true;
	dynamicForceDirection = false;

	setInputDimension(7);
	setTestDimension(7);
	setOutputDimension(3);

}

RotationStrainDataGenerator::~RotationStrainDataGenerator()
{

}

void RotationStrainDataGenerator::generateData()
{
	/*generateDataForPlot();
	return;*/

	origin_data.clear();
	target_data.clear();
	test_origin_data.clear();
	test_target_data.clear();

	std::cout << "generate data start" << std::endl;
	int totaltrainingSize = trainingModal->getNumTrainingHighFreq() + trainingModal->getNumTrainingSet();
	std::cout << "total training size" << totaltrainingSize << std::endl;
	std::cout << "will export type node: " << this->getExportNodetype() << std::endl;

	int numVertex = volumtricMesh->getNumVertices();
	int dataCount = 0;
	int inputdimension = getInputDimension();
	int outputdimension = getOutputDimension();
	VectorXd inputVector(inputdimension);
	VectorXd outputVector(outputdimension);
	for (int i = 0; i < totaltrainingSize; i += 1)
	{ 
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd dq = nq - lq;
		VectorXd w = *(modalrotationMatrix->w_operator)*lq;
		VectorXd e = *(modalrotationMatrix->e_operator)*lq;
		//e.setZero();
		VectorXd g;
		VectorXd f;
		//modalrotationMatrix->computeRotationStrain_To_g_node(w, e, g);
		double poisson_i = trainingModal->getPoissonPerDis(i);

		if (!dynamicForceDirection)
		{
			trainingModal->getTraingLinearForce(i, f);
		}
		else
		{
			trainingModal->getTraingLinearForce(lq, f, poisson_i);
			modalrotationMatrix->computeLocalOrientationMatrixR(modalRotationSparseMatrix, w, false);
			f = *modalRotationSparseMatrix*f;
		}

		updatePotentialOrder(f, -1);

		for (int j = 0; j < numVertex; j++)
		{
			if (ifnodeconstrained[j])
			{
				continue;
			}

			Vector3d nodelq, nodedq;
			for (int k = 0; k < 3; k++)
			{
				nodelq.data()[k] = lq.data()[j * 3 + k];
				nodedq.data()[k] = dq.data()[j * 3 + k];
			}

			Vector3d nodew;
			nodew.data()[0] = w.data()[j * 3 + 0];
			nodew.data()[1] = w.data()[j * 3 + 1];
			nodew.data()[2] = w.data()[j * 3 + 2];

			//do it after all rotated
			Matrix3d alignR;
			Vector3d waxis = nodew.normalized();
			computeNodeAlignRotation(nodelq, waxis, alignR);

			if (!getApplyalignRotation())
			alignR.setIdentity();


			/*Matrix3d nodeF;
			for (int k = 0; k < 9; k++)
			{
				nodeF.data()[k] = g.data()[k+j*9];
			}*/

			//apply align
			//nodeF = alignR*nodeF;
			nodew = alignR*nodew;
			nodedq = alignR*nodedq;
			nodelq = alignR*nodelq;

			/*for (int k = 0; k < 9; k++)
			{
				inputVector.data()[k] = nodeF.data()[k];
			}*/
			double normw = nodew.norm();


			if (std::abs(normw) > 1e-15)
				nodew.normalize();

			double latitude = nodew.dot(Vector3d(0, -1, 0));
			latitude = std::acos(latitude);

			/*for (int k = 0; k < 3; k++)
			{
				inputVector.data()[k] = nodew.data()[k];
			}*/
			inputVector.data()[0] = normw;
			inputVector.data()[1] = latitude;
			//inputVector.data()[2] = nodew.data()[2];


			/*for (int k = 0; k < 6; k++)
			{
			inputVector.data()[k + 3] = e.data()[j * 6 + k];
			}*/

			/*for (int k = 0; k < 3; k++)
			{
			inputVector.data()[3+k] = nodelq.data()[k];
			}*/

			int offset = 2;
			inputVector.data()[offset+0] = nodelq.data()[1];
			inputVector.data()[offset+1] = poisson_i;
			inputVector.data()[offset+2] = node_distance[j];
			inputVector.data()[offset+3] = nodePotentialE[j];
			inputVector.data()[offset+4] = forceAngle[j];

			for (int k = 0; k < 3;k++)
			outputVector.data()[k] = nodedq.data()[k];

			if (dataCount % 8 != 0)
			{
				pushTrainingData(inputVector, outputVector);
			}
			else
			{
				pushTestData(inputVector, outputVector);
			}

			dataCount++;
		}
	}

	std::cout << "export data" << std::endl;

	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "origin_train_" << exportNodetype << ".npy";
	elefilename[0] = data[0].str();

	data[1] << "target_train_" << exportNodetype << ".npy";
	elefilename[1] = data[1].str();

	data[2] << "origin_test_" << exportNodetype << ".npy";
	elefilename[2] = data[2].str();

	data[3] << "target_test_" << exportNodetype << ".npy";
	elefilename[3] = data[3].str();

	trainingModal->exportExampleData(elefilename[0].c_str(), origin_data.data(), inputdimension, origin_data.size());
	trainingModal->exportExampleData(elefilename[1].c_str(), target_data.data(), outputdimension, target_data.size());

	trainingModal->exportExampleData(elefilename[2].c_str(), test_origin_data.data(), inputdimension, test_origin_data.size());
	trainingModal->exportExampleData(elefilename[3].c_str(), test_target_data.data(), outputdimension, test_target_data.size());

	std::cout << "finished export" << std::endl;
}

void RotationStrainDataGenerator::generateDataForPlot()
{
	origin_data.clear();
	target_data.clear();
	test_origin_data.clear();
	test_target_data.clear();

	std::cout << "generate data start" << std::endl;
	int totaltrainingSize = trainingModal->getNumTrainingHighFreq() + trainingModal->getNumTrainingSet();
	std::cout << "total training size" << totaltrainingSize << std::endl;
	std::cout << "will export type node: " << this->getExportNodetype() << std::endl;

	int numVertex = volumtricMesh->getNumVertices();
	int dataCount = 0;
	int inputdimension = 7;
	int outputdimension = 3;
	VectorXd inputVector(inputdimension);
	VectorXd outputVector(outputdimension);
	for (int i = 0; i < totaltrainingSize; i += 20)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd dq = nq - lq;
		VectorXd w = *(modalrotationMatrix->w_operator)*lq;
		VectorXd e = *(modalrotationMatrix->e_operator)*lq;
		//e.setZero();
		VectorXd g;
		VectorXd f;
		//modalrotationMatrix->computeRotationStrain_To_g_node(w, e, g);
		double poisson_i = trainingModal->getPoissonPerDis(i);

		if (!dynamicForceDirection)
		{
			trainingModal->getTraingLinearForce(i, f);
		}
		else
		{
			trainingModal->getTraingLinearForce(lq, f, poisson_i);
			modalrotationMatrix->computeLocalOrientationMatrixR(modalRotationSparseMatrix, w, false);
			f = *modalRotationSparseMatrix*f;
		}

		updatePotentialOrder(f, -1);

		for (int j = 0; j < numVertex; j++)
		{
			if (ifnodeconstrained[j])
			{
				continue;
			}

			Vector3d nodelq, nodedq;
			for (int k = 0; k < 3; k++)
			{
				nodelq.data()[k] = lq.data()[j * 3 + k];
				nodedq.data()[k] = dq.data()[j * 3 + k];
			}

			Vector3d nodew;
			nodew.data()[0] = w.data()[j * 3 + 0];
			nodew.data()[1] = w.data()[j * 3 + 1];
			nodew.data()[2] = w.data()[j * 3 + 2];
			double normw = nodew.norm();
			//do it after all rotated
			Matrix3d alignR;
			Vector3d waxis = nodew.normalized();

			computeNodeAlignRotation(nodelq, waxis, alignR);

			if (!getApplyalignRotation())
				alignR.setIdentity();


			/*Matrix3d nodeF;
			for (int k = 0; k < 9; k++)
			{
			nodeF.data()[k] = g.data()[k+j*9];
			}*/

			//apply align
			//nodeF = alignR*nodeF;
			nodew = alignR*nodew;
			nodedq = alignR*nodedq;
			nodelq = alignR*nodelq;

			/*for (int k = 0; k < 9; k++)
			{
			inputVector.data()[k] = nodeF.data()[k];
			}*/

			if (std::abs(normw) > 1e-15)
				nodew.normalize();

			for (int k = 0; k < 3; k++)
			{
				inputVector.data()[k] = nodew.data()[k];
			}
			inputVector.data()[3] = normw;
			for (int k = 0; k < 3; k++)
			{
				inputVector.data()[4 + k] = nodelq.data()[k];
			}


			/*for (int k = 0; k < 6; k++)
			{
			inputVector.data()[k + 3] = e.data()[j * 6 + k];
			}*/

			/*for (int k = 0; k < 3; k++)
			{
			inputVector.data()[3+k] = nodelq.data()[k];
			}*/


			for (int k = 0; k < 3; k++)
				outputVector.data()[k] = nodedq.data()[k];

			if (dataCount % 8 != 0)
			{
				pushTrainingData(inputVector, outputVector);
			}
			else
			{
				pushTestData(inputVector, outputVector);
			}

			dataCount++;
		}
	}

	std::cout << "export data" << std::endl;

	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "origin_train_" << 0 << ".txt";
	elefilename[0] = data[0].str();

	data[1] << "target_train_" << 0 << ".txt";
	elefilename[1] = data[1].str();

	data[2] << "origin_test_" << 0 << ".txt";
	elefilename[2] = data[2].str();

	data[3] << "target_test_" << 0 << ".txt";
	elefilename[3] = data[3].str();

	trainingModal->exportExampleDataAscii(elefilename[0].c_str(), origin_data.data(), inputdimension, origin_data.size());
	trainingModal->exportExampleDataAscii(elefilename[1].c_str(), target_data.data(), outputdimension, target_data.size());

	trainingModal->exportExampleDataAscii(elefilename[2].c_str(), test_origin_data.data(), inputdimension, test_origin_data.size());
	trainingModal->exportExampleDataAscii(elefilename[3].c_str(), test_target_data.data(), outputdimension, test_target_data.size());

	std::cout << "finished export" << std::endl;
}

void RotationStrainDataGenerator::createInputData(VectorXd &lq, VectorXd &dNNInput)
{
	VectorXd w = *(modalrotationMatrix->w_operator)*lq;
	VectorXd e = *(modalrotationMatrix->e_operator)*lq;
	VectorXd g;
	modalrotationMatrix->computeRotationStrain_To_g_node(w, e, g);
	LoboVolumetricMesh::Material* materia = volumtricMesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	double poisson = enmateria->getNu();
	int dimension = 16;
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);



	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodelq;
		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		Vector3d nodew;
		nodew.data()[0] = w.data()[j * 3 + 0];
		nodew.data()[1] = w.data()[j * 3 + 1];
		nodew.data()[2] = w.data()[j * 3 + 2];

		//do it after all rotated
		Matrix3d alignR;
		Vector3d waxis = nodew.normalized();
		computeNodeAlignRotation(nodelq, waxis, alignR);

		if (!getApplyalignRotation())
			alignR.setIdentity();

		alignRList[j] = alignR;
		
		Matrix3d nodeF;
		for (int k = 0; k < 9; k++)
		{
			nodeF.data()[k] = g.data()[k + j * 9];
		}

		//apply align
		nodew = alignR*nodew;
		nodelq = alignR*nodelq;

		/*for (int k = 0; k < 9; k++)
		{
		dNNInput.data()[k+j*dimension] = nodeF.data()[k];
		}
		*/
		for (int k = 0; k < 3; k++)
		{
			dNNInput.data()[k + j*dimension] = nodew.data()[k];
		}

		for (int k = 0; k < 6; k++)
		{
			dNNInput.data()[k + 3 + j*dimension] = e.data()[j * 6 + k];
		}
		
		for (int k = 0; k < 3;k++)
		dNNInput.data()[9+k + j*dimension] = nodelq.data()[k];

		dNNInput.data()[12 + j*dimension] = poisson;

		dNNInput.data()[13 + j*dimension] = node_distance[j];


		dNNInput.data()[14 + j*dimension] = nodePotentialE[j];
		dNNInput.data()[15 + j*dimension] = forceAngle[j];

	}
}

void RotationStrainDataGenerator::createInputData(VectorXd &lq, VectorXd &dNNInput, VectorXd &force)
{
	VectorXd w = *(modalrotationMatrix->w_operator)*lq;
	VectorXd e = *(modalrotationMatrix->e_operator)*lq;
	//e.setZero();
	VectorXd g;
	//modalrotationMatrix->computeRotationStrain_To_g_node(w, e, g);
	LoboVolumetricMesh::Material* materia = volumtricMesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	double poisson = enmateria->getNu();
	int dimension = getInputDimension();
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);

	updatePotentialOrder(force, -1);

	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodelq;
		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		Vector3d nodew;
		nodew.data()[0] = w.data()[j * 3 + 0];
		nodew.data()[1] = w.data()[j * 3 + 1];
		nodew.data()[2] = w.data()[j * 3 + 2];

		//do it after all rotated
		Matrix3d alignR;
		Vector3d waxis = nodew.normalized();
		computeNodeAlignRotation(nodelq, waxis, alignR);

		if (!getApplyalignRotation())
			alignR.setIdentity();

		alignRList[j] = alignR;

		//Matrix3d nodeF;
		//for (int k = 0; k < 9; k++)
		//{
		//	nodeF.data()[k] = g.data()[k + j * 9];
		//}

		//apply align
		nodew = alignR*nodew;
		nodelq = alignR*nodelq;
		//nodeF = alignR*nodeF;

		/*for (int k = 0; k < 9; k++)
		{
		dNNInput.data()[k+j*dimension] = nodeF.data()[k];
		}*/
		double normw = nodew.norm();

		if (std::abs(normw) > 1e-15)
			nodew.normalize();

		double latitude = nodew.dot(Vector3d(0, -1, 0));
		latitude = std::acos(latitude);
		/*for (int k = 0; k < 3; k++)
		{
		dNNInput.data()[k + j*dimension] = nodew.data()[k];
		}*/
		dNNInput.data()[0 + j*dimension] = normw;
		dNNInput.data()[1 + j*dimension] = latitude;
		//dNNInput.data()[2 + j*dimension] = nodew.data()[2];

		
		//dNNInput.data()[2 + j*dimension] = nodew.data()[2];



		/*for (int k = 0; k < 6; k++)
		{
		dNNInput.data()[k + 3 + j*dimension] = e.data()[j * 6 + k];
		}*/
		
		/*for (int k = 0; k < 3; k++)
			dNNInput.data()[9-6 + k + j*dimension] = nodelq.data()[k];*/
		int offest = 2;
		dNNInput.data()[offest + j*dimension] = nodelq.data()[1];
		dNNInput.data()[offest+1 + j*dimension] = poisson;
		dNNInput.data()[offest+2 + j*dimension] = node_distance[j];
		dNNInput.data()[offest+3 + j*dimension] = nodePotentialE[j];
		dNNInput.data()[offest+4 + j*dimension] = forceAngle[j];
	}
}

void RotationStrainDataGenerator::createInputDataWithRotation(VectorXd &lq, VectorXd &dNNInput, VectorXd &force)
{
	VectorXd w = *(modalrotationMatrix->w_operator)*lq;
	VectorXd e = *(modalrotationMatrix->e_operator)*lq;
	//e.setZero();
	VectorXd g;
	//modalrotationMatrix->computeRotationStrain_To_g_node(w, e, g);
	LoboVolumetricMesh::Material* materia = volumtricMesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	double poisson = enmateria->getNu();
	int dimension = getInputDimension()+9;
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);

	updatePotentialOrder(force, -1);

	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodelq;
		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		Vector3d nodew;
		nodew.data()[0] = w.data()[j * 3 + 0];
		nodew.data()[1] = w.data()[j * 3 + 1];
		nodew.data()[2] = w.data()[j * 3 + 2];

		//do it after all rotated
		Matrix3d alignR;
		Vector3d waxis = nodew.normalized();
		computeNodeAlignRotation(nodelq, waxis, alignR);

		if (!getApplyalignRotation())
			alignR.setIdentity();

		alignRList[j] = alignR;

		//Matrix3d nodeF;
		//for (int k = 0; k < 9; k++)
		//{
		//	nodeF.data()[k] = g.data()[k + j * 9];
		//}

		//apply align
		nodew = alignR*nodew;
		nodelq = alignR*nodelq;
		//nodeF = alignR*nodeF;

		/*for (int k = 0; k < 9; k++)
		{
		dNNInput.data()[k+j*dimension] = nodeF.data()[k];
		}*/
		double normw = nodew.norm();

		if (std::abs(normw) > 1e-15)
			nodew.normalize();

		double latitude = nodew.dot(Vector3d(0, -1, 0));
		latitude = std::acos(latitude);
		/*for (int k = 0; k < 3; k++)
		{
		dNNInput.data()[k + j*dimension] = nodew.data()[k];
		}*/
		dNNInput.data()[0 + j*dimension] = normw;
		dNNInput.data()[1 + j*dimension] = latitude;
		//dNNInput.data()[2 + j*dimension] = nodew.data()[2];


		//dNNInput.data()[2 + j*dimension] = nodew.data()[2];



		/*for (int k = 0; k < 6; k++)
		{
		dNNInput.data()[k + 3 + j*dimension] = e.data()[j * 6 + k];
		}*/

		/*for (int k = 0; k < 3; k++)
		dNNInput.data()[9-6 + k + j*dimension] = nodelq.data()[k];*/
		int offest = 2;
		dNNInput.data()[offest + j*dimension] = nodelq.data()[1];
		dNNInput.data()[offest + 1 + j*dimension] = poisson;
		dNNInput.data()[offest + 2 + j*dimension] = node_distance[j];
		dNNInput.data()[offest + 3 + j*dimension] = nodePotentialE[j];
		dNNInput.data()[offest + 4 + j*dimension] = forceAngle[j];

		for (int k = 0; k < 9; k++)
		{
			dNNInput.data()[offest + 5 + k + j*dimension] = alignR.data()[k];
		}
	}
}

void RotationStrainDataGenerator::updatePotentialOrder(VectorXd &exforce, int idnex /*= -1*/)
{
	int numVertex = volumtricMesh->getNumVertices();
	Vector3d extforcedirection;

	extforcedirection.setZero();
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d noforce;
		noforce.data()[0] = exforce.data()[i * 3 + 0];
		noforce.data()[1] = exforce.data()[i * 3 + 1];
		noforce.data()[2] = exforce.data()[i * 3 + 2];

		if (noforce.norm() > 1e-14)
			noforce.normalize();
		else
			noforce.setZero();

		extforcedirection.data()[0] += noforce.data()[0];
		extforcedirection.data()[1] += noforce.data()[1];
		extforcedirection.data()[2] += noforce.data()[2];
	}

	if (extforcedirection.norm()>1e-15)
	extforcedirection.normalize();
	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodeforce;

		if (getForcefieldType() == 1)
		{
			/*nodeforce = volumtricMesh->getNodeRestPosition(j);
			nodeforce -= forcefield->centeraxis_position;
			nodeforce = nodeforce - nodeforce.dot(axis)*axis;*/

			nodeforce = distanceToZeroPotential[j];
			nodeforce = nodeforce - nodeforce.dot(extforcedirection)*extforcedirection;
			nodePotentialE[j] = nodeforce.norm();

			//nodePotentialE[j] = -1;
			forceAngle[j] = -1;
		}
		else if (getForcefieldType() == 0)
		{
			nodePotentialE[j] = distanceToZeroPotential[j].dot(extforcedirection);

			forceAngle[j] = nodePotentialE[j] / distanceToZeroPotential[j].norm();

			//hard code for multi cluster constraints
				/*Vector3d nodep = volumtricMesh->getNodeRestPosition(j);
				Vector3d zeropoint;
				if (nodep.data()[0] < 0)
				{
				zeropoint = nodep - Vector3d(-0.5, 0, 0);
				}
				else
				{
				zeropoint = nodep - Vector3d(0.5, 0, 0);
				}
				forceAngle[j] = zeropoint.dot(extforcedirection) / zeropoint.norm();*/
		}
	}

	/*if (getForcefieldType() == 1)
	{
		return;
	}*/

	double min = nodePotentialE.minCoeff();
	double max = nodePotentialE.maxCoeff();
	double scale = max - min;

	for (int j = 0; j < numVertex; j++)
	{
		nodePotentialE[j] -= min;
	}
	nodePotentialE /= scale;

}

void RotationStrainDataGenerator::pushTrainingData(VectorXd &input, VectorXd &output)
{
	for (int i = 0; i < input.rows(); i++)
	{
		origin_data.push_back(input.data()[i]);
	}

	for (int i = 0; i < output.rows(); i++)
	{
		target_data.push_back(output.data()[i]);
	}
}

void RotationStrainDataGenerator::pushTestData(VectorXd &input, VectorXd &output)
{
	for (int i = 0; i < input.rows(); i++)
	{
		test_origin_data.push_back(input.data()[i]);
	}

	for (int i = 0; i < output.rows(); i++)
	{
		test_target_data.push_back(output.data()[i]);
	}
}
