#include "FullfeatureDataGenerator.h"


#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"
#include "Functions/GeoMatrix.h"
#include <fstream>
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"
#include "Simulator/ForceField/RotateForceField.h"

FullfeatureDataGenerator::FullfeatureDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/) :RotationStrainDataGenerator(tetmeshvox_, cubemesh_, volumtricMesh_, trainingModal_, modalrotaionMatrix_, modalRotationSparseMatrix_, numConstrainedDOFs, constrainedDOFs)
{
	setInputDimension(11);
	setTestDimension(11);
	setOutputDimension(3);
	applyalignRotation = false;

}

FullfeatureDataGenerator::~FullfeatureDataGenerator()
{

}

void FullfeatureDataGenerator::generateData()
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

			inputVector.data()[7] = poisson_i;
			inputVector.data()[8] = node_distance[j];
			inputVector.data()[9] = nodePotentialE[j];
			inputVector.data()[10] = forceAngle[j];

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

void FullfeatureDataGenerator::createInputData(VectorXd &lq, VectorXd &dNNInput, VectorXd &force)
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

		for (int k = 0; k < 3; k++)
		{
			dNNInput.data()[k + j*dimension] = nodew.data()[k];
		}
		dNNInput.data()[3 + j*dimension] = normw;
		for (int k = 0; k < 3; k++)
		{
			dNNInput.data()[4+k + j*dimension] = nodelq.data()[k];
		}
		/*for (int k = 0; k < 6; k++)
		{
		dNNInput.data()[k + 3 + j*dimension] = e.data()[j * 6 + k];
		}*/

		/*for (int k = 0; k < 3; k++)
		dNNInput.data()[9-6 + k + j*dimension] = nodelq.data()[k];*/
		dNNInput.data()[7 + j*dimension] = poisson;
		dNNInput.data()[8 + j*dimension] = node_distance[j];
		dNNInput.data()[9 + j*dimension] = nodePotentialE[j];
		dNNInput.data()[10 + j*dimension] = forceAngle[j];
	}
}
