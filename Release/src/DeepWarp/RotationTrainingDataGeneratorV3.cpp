#include "RotationTrainingDataGeneratorV3.h"
#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"
#include "Functions/GeoMatrix.h"
#include <fstream>
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"
#include <QElapsedTimer>


RotationTrainingDataGeneratorV3::RotationTrainingDataGeneratorV3(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/) :RotationTrainingDataGeneratorV2(tetmeshvox_,cubemesh_,volumtricMesh_,trainingModal_,modalrotaionMatrix_,modalRotationSparseMatrix_,numConstrainedDOFs,constrainedDOFs)
{
	alignRList.resize(this->volumtricMesh->getNumVertices());
	nodePotentialE.resize(this->volumtricMesh->getNumVertices());
	nodePotentialE.setZero();
	forceAngle.resize(this->volumtricMesh->getNumVertices());
	forceAngle.setZero();
	nodeForceCoordinate.resize(this->volumtricMesh->getNumVertices() * 3);
	nodeForceCoordinate.setZero();

	int numVertex = volumtricMesh->getNumVertices();
	nodeMass.resize(numVertex);
	volumtricMesh->computeNodeVolume(nodeMass.data());
	for (int i = 0; i < numVertex; i++)
	{
		double density = volumtricMesh->getNodeMaterial(i)->getDensity();
		nodeMass[i] *= density;
	}
	forcefieldType = 0; //0 gravity 1 rotated force

	if (trainingModal!=NULL)
	forcefieldType = trainingModal->getForcefieldType();

}

RotationTrainingDataGeneratorV3::~RotationTrainingDataGeneratorV3()
{

}

void RotationTrainingDataGeneratorV3::generateData()
{
	std::cout << "generate data start" << std::endl;
	int totaltrainingSize = trainingModal->getNumTrainingHighFreq() + trainingModal->getNumTrainingSet();
	std::cout << "total training size" << totaltrainingSize << std::endl;
	std::cout << "will export type node: " << this->getExportNodetype() << std::endl;

	std::vector<double> origin_data;
	std::vector<double> target_data;

	std::vector<double> test_origin_data;
	std::vector<double> test_target_data;

	int numVertex = volumtricMesh->getNumVertices();

	int count = 0;
	int totalData = totaltrainingSize*numVertex;

	std::vector<bool> ifnodeconstrained(numVertex);
	std::fill(ifnodeconstrained.begin(), ifnodeconstrained.end(), false);

	for (int i = 0; i < numConstrainedDOFs / 3; i++)
	{
		int nodeindex = constrainedDOFs[i * 3] / 3;
		ifnodeconstrained[nodeindex] = true;
	}
	std::vector<double> data_weight;
	int dataCount = 0;
	for (int i = 0; i < totaltrainingSize; i += 1)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd exforce;
		double poisson_i = trainingModal->getPoissonPerDis(i);
		//poisson_i discribe
		poisson_i = (poisson_i - 0.2) / (0.45 - 0.2);
		
		trainingModal->getTraingLinearForce(i, exforce);
		//trainingModal->getTraingLinearForce(lq, exforce);

		VectorXd w = (*modalRotationSparseMatrix)*lq;
		double nqNorm = nq.norm();

		//modalrotationMatrix->computeLocalOrientationMatrixR(rotationSparseMatrixR, w, false);
		//exforce = *rotationSparseMatrixR*exforce;
		
		/*modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);
		nq = (*rotationSparseMatrixR)*lq;*/

		VectorXd dq = nq-lq;
		//trainingModal->setTrainingNonLinearDis(dq, i);

		volumtricMesh->setDisplacement(lq.data());

		VectorXd Estrain;
		modalrotationMatrix->computeModalStrainMatrix(Estrain);
		updatePotentialOrder(exforce,i);

		std::vector<Vector4d> w_list(numVertex);
		for (int j = 0; j < numVertex; j++)
		{
			//Matrix3d R_j;
			//volumtricMesh->computeNodeRotationRing(j, R_j);

			Vector3d w_i;
			for (int k = 0; k < 3; k++)
				w_i.data()[k] = w.data()[j * 3 + k];
			double angle = w_i.norm();

			if (std::abs(angle) > 1e-15)
				w_i.normalize();

			for (int k = 0; k < 3; k++)
				w_list[j][k] = w_i[k];
			w_list[j][3] = angle;

			//w_list[j] = convertRtoAxisAngle(R_j);
		}

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

			std::vector<Vector4d> nodeW(1);
			nodeW[0] = w_list[j];

			//do it after all rotated
			Matrix3d alignR;

			Vector3d waxis;
			waxis.data()[0] = nodeW[0].data()[0];
			waxis.data()[1] = nodeW[0].data()[1];
			waxis.data()[2] = nodeW[0].data()[2];

			computeNodeAlignRotation(nodelq, waxis, alignR);


			AngleAxisd node_aa;
			node_aa = alignR;
			Vector4d node_alignAxisAngle;

			node_alignAxisAngle.data()[0] = node_aa.axis().data()[0];
			node_alignAxisAngle.data()[1] = node_aa.axis().data()[1];
			node_alignAxisAngle.data()[2] = node_aa.axis().data()[2];
			node_alignAxisAngle.data()[3] = node_aa.angle();

			//don't apply this 
			//if (0)
			{
				nodelq = alignR*nodelq;
				nodedq = alignR*nodedq;

				for (int k = 0; k < nodeW.size(); k++)
				{
					Vector3d temp;
					for (int l = 0; l < 3; l++)
						temp.data()[l] = nodeW[k].data()[l];

					temp = alignR*temp;

					temp.data()[0] = poisson_i;

					for (int l = 0; l < 3; l++)
						nodeW[k].data()[l] = temp.data()[l];
				}
			}
			
			//nodedq.data()[2] = 0;

			int scaleNum = 1;
			Vector3d originlq = nodelq;
			Vector3d origindq = nodedq;
			while (scaleNum)
			{
				nodelq = originlq* 1.0 / scaleNum;
				nodedq = origindq* 1.0 / scaleNum;

				if (dataCount % 5 != 0)
				{
					//only need y 
					
					origin_data.push_back(nodelq.data()[1]);
					/*for (int k = 0; k < 6; k++)
						origin_data.push_back(Estrain[j * 6 + k]);*/

					for (int k = 0; k < 3; k++)
					{
						target_data.push_back(nodedq.data()[k]);
					}

					for (int k = 0; k < 9; k++)
					{
						//target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < nodeW.size(); k++)
					{
						for (int l = 0; l < 4; l++)
						{
							origin_data.push_back(nodeW[k].data()[l]);
						}
						
					}

					origin_data.push_back(nodePotentialE[j]);
					origin_data.push_back(forceAngle[j]);

					origin_data.push_back(node_distance[j]);



					for (int k = 0; k < 4; k++)
					{
						//origin_data.push_back(node_alignAxisAngle.data()[k]);
					}

					data_weight.push_back(nqNorm);

				}
				else
				{
					test_origin_data.push_back(nodelq.data()[1]);
					/*for (int k = 0; k < 6; k++)
						test_origin_data.push_back(Estrain[j * 6 + k]);*/

					for (int k = 0; k < 3; k++)
					{
						test_target_data.push_back(nodedq.data()[k]);
					}

					for (int k = 0; k < 9; k++)
					{
						//test_target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < nodeW.size(); k++)
					{
						for (int l = 0; l < 4; l++)
						{
							test_origin_data.push_back(nodeW[k].data()[l]);
						}
					}
					test_origin_data.push_back(nodePotentialE[j]);
					test_origin_data.push_back(forceAngle[j]);

					test_origin_data.push_back(node_distance[j]);

					for (int k = 0; k < 4; k++)
					{
						//test_origin_data.push_back(node_alignAxisAngle.data()[k]);
					}
				}
				scaleNum--;
			}
		}
		dataCount++;
	}

	int dimension = 4 + 2 +1+1;
	int outputdimension = 3;
	std::cout << "export data" << std::endl;

	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "origin_train_" << exportNodetype << ".txt";
	elefilename[0] = data[0].str();

	data[1] << "target_train_" << exportNodetype << ".txt";
	elefilename[1] = data[1].str();

	data[2] << "origin_test_" << exportNodetype << ".txt";
	elefilename[2] = data[2].str();

	data[3] << "target_test_" << exportNodetype << ".txt";
	elefilename[3] = data[3].str();

	trainingModal->exportExampleData(elefilename[0].c_str(), origin_data.data(), dimension, origin_data.size());
	trainingModal->exportExampleData(elefilename[1].c_str(), target_data.data(), outputdimension, target_data.size());

	trainingModal->exportExampleData(elefilename[2].c_str(), test_origin_data.data(), dimension, test_origin_data.size());
	trainingModal->exportExampleData(elefilename[3].c_str(), test_target_data.data(), outputdimension, test_target_data.size());

	trainingModal->exportExampleData("data_weight.txt", data_weight.data(), 1, data_weight.size());

	std::cout << "finished export" << std::endl;

}

void RotationTrainingDataGeneratorV3::testDataByDNN(LoboNeuralNetwork* loboNeuralNetwork)
{
	/*testDataByDNNUserDefine(loboNeuralNetwork);
	return;*/
	std::cout << "generate data start" << std::endl;
	int totaltrainingSize = trainingModal->getNumTrainingHighFreq() + trainingModal->getNumTrainingSet();
	std::cout << "total training size" << totaltrainingSize << std::endl;
	std::cout << "will export type node: " << this->getExportNodetype() << std::endl;

	std::vector<double> origin_data;
	std::vector<double> target_data;

	std::vector<double> test_origin_data;
	std::vector<double> test_target_data;

	int numVertex = volumtricMesh->getNumVertices();

	int count = 0;
	int totalData = totaltrainingSize*numVertex;

	std::vector<bool> ifnodeconstrained(numVertex);
	std::fill(ifnodeconstrained.begin(), ifnodeconstrained.end(), false);

	for (int i = 0; i < numConstrainedDOFs / 3; i++)
	{
		int nodeindex = constrainedDOFs[i * 3] / 3;
		ifnodeconstrained[nodeindex] = true;
	}
	std::vector<double> data_weight;
	int dataCount = 0;

	double testError = 0;
	int numTestData = 0;
	int pickedNodes = 0;

	for (int i = 0; i < totaltrainingSize; i += 1)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd w = (*modalRotationSparseMatrix)*lq;

		/*modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);
		nq = (*rotationSparseMatrixR)*lq;*/

		VectorXd exforce;
		trainingModal->getTraingLinearForce(i, exforce);
		//trainingModal->getTraingLinearForce(lq, exforce);
		double poisson_i = trainingModal->getPoissonPerDis(i);
		//poisson_i discribe
		poisson_i = (poisson_i - 0.2) / (0.45 - 0.2);
		//modalrotationMatrix->computeLocalOrientationMatrixR(rotationSparseMatrixR, w, false);
		//exforce = *rotationSparseMatrixR*exforce;

		updatePotentialOrder(exforce, i);

		double nqNorm = nq.norm();

		VectorXd dq = nq - lq;
		//trainingModal->setTrainingNonLinearDis(dq, i);

		volumtricMesh->setDisplacement(lq.data());

		VectorXd Estrain;
		modalrotationMatrix->computeModalStrainMatrix(Estrain);
		

		std::vector<Vector4d> w_list(numVertex);
		for (int j = 0; j < numVertex; j++)
		{
			Matrix3d R_j;
			volumtricMesh->computeNodeRotationRing(j, R_j);


			Vector3d w_i;
			for (int k = 0; k < 3; k++)
				w_i.data()[k] = w.data()[j * 3 + k];
			double angle = w_i.norm();

			if (std::abs(angle)>1e-15)
				w_i.normalize();

			for (int k = 0; k < 3; k++)
				w_list[j][k] = w_i[k];
			w_list[j][3] = angle;

			//w_list[j] = convertRtoAxisAngle(R_j);
		}

		pickedNodes = 0;
		for (int j = 0; j < numVertex; j++)
		{
			if (ifnodeconstrained[j])
			{
				continue;
			}
			
			/*if (!(j == 330))
			{
			continue;
			}*/

			/*if (node_distance[j] > 0.6||node_distance[j]<0.5)
			{
				continue;
			}*/

			pickedNodes++;
			Vector3d nodelq, nodedq;

			for (int k = 0; k < 3; k++)
			{
				nodelq.data()[k] = lq.data()[j * 3 + k];
				nodedq.data()[k] = dq.data()[j * 3 + k];
			}

			std::vector<Vector4d> nodeW(1);
			nodeW[0] = w_list[j];

			//do it after all rotated
			Matrix3d alignR;

			Vector3d waxis;
			waxis.data()[0] = nodeW[0].data()[0];
			waxis.data()[1] = nodeW[0].data()[1];
			waxis.data()[2] = nodeW[0].data()[2];

			computeNodeAlignRotation(nodelq, waxis, alignR);

			AngleAxisd node_aa;
			node_aa = alignR;
			Vector4d node_alignAxisAngle;

			node_alignAxisAngle.data()[0] = node_aa.axis().data()[0];
			node_alignAxisAngle.data()[1] = node_aa.axis().data()[1];
			node_alignAxisAngle.data()[2] = node_aa.axis().data()[2];
			node_alignAxisAngle.data()[3] = node_aa.angle();

			//don't apply this 
			//if (0)
			{
				nodelq = alignR*nodelq;
				nodedq = alignR*nodedq;

				for (int k = 0; k < nodeW.size(); k++)
				{
					Vector3d temp;
					for (int l = 0; l < 3; l++)
						temp.data()[l] = nodeW[k].data()[l];

					temp = alignR*temp;

					temp.data()[0] = poisson_i;

					for (int l = 0; l < 3; l++)
						nodeW[k].data()[l] = temp.data()[l];
				}
			}

			//nodedq.data()[2] = 0;


			int scaleNum = 1;
			Vector3d originlq = nodelq;
			Vector3d origindq = nodedq;
			while (scaleNum)
			{
				nodelq = originlq* 1.0 / scaleNum;
				nodedq = origindq* 1.0 / scaleNum;
				int dimension = 8;
				if (dataCount % 5 != 0)
				{
					//only need y 
					origin_data.push_back(nodelq.data()[1]);
					
					/*for (int k = 0; k < 6;k++)
					origin_data.push_back(Estrain[j*6+k]);*/

					

					for (int k = 0; k < 9; k++)
					{
						//target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < nodeW.size(); k++)
					{
						for (int l = 0; l < 4; l++)
						{
							origin_data.push_back(nodeW[k].data()[l]);
						}
					}

					origin_data.push_back(nodePotentialE[j]);

					origin_data.push_back(forceAngle[j]);
					origin_data.push_back(node_distance[j]);


					VectorXd inputv(dimension);
					for (int i = 0; i < dimension; i++)
					{
						inputv[i] = origin_data.data()[origin_data.size() - dimension + i];
					}

					//nodedq = loboNeuralNetwork->predictV2(inputv);
					Vector3d predictedvalue = loboNeuralNetwork->predictV2(inputv);

					testError += ((nodedq - predictedvalue).norm() / nodedq.norm());
					numTestData++;

					for (int k = 0; k < 3; k++)
					{
						target_data.push_back(nodedq.data()[k]);
					}

					for (int k = 0; k < 4; k++)
					{
						//origin_data.push_back(node_alignAxisAngle.data()[k]);
					}

					data_weight.push_back(nqNorm);

				}
				else
				{
					test_origin_data.push_back(nodelq.data()[1]);
					/*for (int k = 0; k < 6; k++)
						test_origin_data.push_back(Estrain[j * 6 + k]);*/


					for (int k = 0; k < nodeW.size(); k++)
					{
						for (int l = 0; l < 4; l++)
						{
							test_origin_data.push_back(nodeW[k].data()[l]);
						}
					}
					test_origin_data.push_back(nodePotentialE[j]);

					test_origin_data.push_back(forceAngle[j]);
					test_origin_data.push_back(node_distance[j]);


					VectorXd inputv(dimension);
					for (int i = 0; i < dimension; i++)
					{
						inputv[i] = test_origin_data.data()[test_origin_data.size() - dimension + i];
					}

					//test 
					nodedq = loboNeuralNetwork->predictV2(inputv);

					for (int k = 0; k < 3; k++)
					{
						test_target_data.push_back(nodedq.data()[k]);
					}
				}
				scaleNum--;
			}

			nodedq = alignR.transpose()*nodedq;
			for (int k = 0; k < 3; k++)
			{
				dq.data()[j * 3 + k] = nodedq.data()[k];
			}
		}
		trainingModal->setTrainingNonLinearDis(dq+lq, i);
		dataCount++;
	}

	std::cout << "pickedNodes " << pickedNodes << std::endl;

	int dimension = 4 + 2+2;
	int outputdimension = 3;
	std::cout << "export data" << std::endl;

	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "origin_train_" << exportNodetype << ".txt";
	elefilename[0] = data[0].str();

	data[1] << "target_train_" << exportNodetype << ".txt";
	elefilename[1] = data[1].str();

	data[2] << "origin_test_" << exportNodetype << ".txt";
	elefilename[2] = data[2].str();

	data[3] << "target_test_" << exportNodetype << ".txt";
	elefilename[3] = data[3].str();


	trainingModal->exportExampleData(elefilename[0].c_str(), origin_data.data(), dimension, origin_data.size());
	trainingModal->exportExampleData(elefilename[1].c_str(), target_data.data(), outputdimension, target_data.size());

	trainingModal->exportExampleData(elefilename[2].c_str(), test_origin_data.data(), dimension, test_origin_data.size());
	trainingModal->exportExampleData(elefilename[3].c_str(), test_target_data.data(), outputdimension, test_target_data.size());

	trainingModal->exportExampleData("data_weight.txt", data_weight.data(), 1, data_weight.size());

	std::cout << "test error" << testError / numTestData << std::endl;

	std::cout << "finished export" << std::endl;
}

void RotationTrainingDataGeneratorV3::createInputData(VectorXd &lq, VectorXd &dNNInput)
{
	int dimension = 8;
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);

	std::vector<Matrix3d> R_list(numVertex);
	std::vector<Vector4d> w_list(numVertex);

	VectorXd w = (*modalRotationSparseMatrix)*lq;
	volumtricMesh->setDisplacement(lq.data());

	VectorXd exforce;
	trainingModal->getTraingLinearForce(lq, exforce);
	updatePotentialOrder(exforce);

	//VectorXd Estrain;
	//modalrotationMatrix->computeModalStrainMatrix(Estrain);
	for (int j = 0; j < numVertex; j++)
	{
		//w_list[j] = convertRtoAxisAngle(R_list[j]);
		Matrix3d R_j;
		volumtricMesh->computeNodeRotationRing(j, R_j);
		Vector3d w_i;
		for (int k = 0; k < 3; k++)
			w_i.data()[k] = w.data()[j * 3 + k];
		double angle = w_i.norm();
		if (std::abs(angle) > 1e-15)
			w_i.normalize();
		for (int k = 0; k < 3; k++)
			w_list[j][k] = w_i[k];
		w_list[j][3] = angle;
		//w_list[j] = convertRtoAxisAngle(R_j);
	}

	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodelq, nodedq;

		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		std::vector<Vector4d> nodeW(1);
		nodeW[0] = w_list[j];

		Matrix3d alignR;
		Vector3d waxis;
		waxis.data()[0] = nodeW[0].data()[0];
		waxis.data()[1] = nodeW[0].data()[1];
		waxis.data()[2] = nodeW[0].data()[2];

		computeNodeAlignRotation(nodelq, waxis, alignR);

		alignRList[j] = alignR;

		AngleAxisd node_aa;
		node_aa = alignR;
		Vector4d node_alignAxisAngle;

		node_alignAxisAngle.data()[0] = node_aa.axis().data()[0];
		node_alignAxisAngle.data()[1] = node_aa.axis().data()[1];
		node_alignAxisAngle.data()[2] = node_aa.axis().data()[2];
		node_alignAxisAngle.data()[3] = node_aa.angle();


		nodelq = alignR*nodelq;
		for (int k = 0; k < nodeW.size(); k++)
		{
			Vector3d temp;
			for (int l = 0; l < 3; l++)
				temp.data()[l] = nodeW[k].data()[l];
			temp = alignR*temp;
			for (int l = 0; l < 3; l++)
				nodeW[k].data()[l] = temp.data()[l];
		}

		dNNInput.data()[j*dimension + 0] = nodelq.data()[1];
		/*for (int k = 0; k < 6; k++)
			dNNInput.data()[j*dimension + 2+k] = Estrain[j * 6 + k];*/

		
		dNNInput.data()[j*dimension + 1] = nodeW[0].data()[0];
		dNNInput.data()[j*dimension + 2] = nodeW[0].data()[1];
		dNNInput.data()[j*dimension + 3] = nodeW[0].data()[2];
		dNNInput.data()[j*dimension + 4] = nodeW[0].data()[3];
		dNNInput.data()[j*dimension + 5] = nodePotentialE[j];
		dNNInput.data()[j*dimension + 6] = forceAngle[j];

		dNNInput.data()[j*dimension + 7] = node_distance[j];

	}
}

void RotationTrainingDataGeneratorV3::createInputData(VectorXd &lq, VectorXd &dNNInput, VectorXd filedforce)
{
	LoboVolumetricMesh::Material* materia = volumtricMesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	double poisson = enmateria->getNu();
	poisson = (poisson - 0.2) / (0.45 - 0.2);

	int dimension = 8;
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);

	std::vector<Matrix3d> R_list(numVertex);
	std::vector<Vector4d> w_list(numVertex);

	VectorXd w = (*modalRotationSparseMatrix)*lq;
	//volumtricMesh->setDisplacement(lq.data());

	updatePotentialOrder(filedforce);

	//VectorXd Estrain;
	//modalrotationMatrix->computeModalStrainMatrix(Estrain);
	for (int j = 0; j < numVertex; j++)
	{
		//w_list[j] = convertRtoAxisAngle(R_list[j]);
		//Matrix3d R_j;
		//volumtricMesh->computeNodeRotationRing(j, R_j);
		
		Vector3d w_i;
		for (int k = 0; k < 3; k++)
			w_i.data()[k] = w.data()[j * 3 + k];
		double angle = w_i.norm();
		if (std::abs(angle) > 1e-15)
			w_i.normalize();

		for (int k = 0; k < 3; k++)
			w_list[j][k] = w_i[k];
		w_list[j][3] = angle;

		//w_list[j] = convertRtoAxisAngle(R_j);
	}

	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodelq, nodedq;

		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		std::vector<Vector4d> nodeW(1);
		nodeW[0] = w_list[j];

		Matrix3d alignR;
		Vector3d waxis;
		waxis.data()[0] = nodeW[0].data()[0];
		waxis.data()[1] = nodeW[0].data()[1];
		waxis.data()[2] = nodeW[0].data()[2];

		computeNodeAlignRotation(nodelq, waxis, alignR);

		alignRList[j] = alignR;

		AngleAxisd node_aa;
		node_aa = alignR;
		Vector4d node_alignAxisAngle;

		node_alignAxisAngle.data()[0] = node_aa.axis().data()[0];
		node_alignAxisAngle.data()[1] = node_aa.axis().data()[1];
		node_alignAxisAngle.data()[2] = node_aa.axis().data()[2];
		node_alignAxisAngle.data()[3] = node_aa.angle();


		nodelq = alignR*nodelq;
		for (int k = 0; k < nodeW.size(); k++)
		{
			Vector3d temp;
			for (int l = 0; l < 3; l++)
				temp.data()[l] = nodeW[k].data()[l];
			temp = alignR*temp;
			
			temp.data()[0] = poisson;

			for (int l = 0; l < 3; l++)
				nodeW[k].data()[l] = temp.data()[l];
		}

		dNNInput.data()[j*dimension + 0] = nodelq.data()[1];
		/*for (int k = 0; k < 6; k++)
		dNNInput.data()[j*dimension + 2+k] = Estrain[j * 6 + k];*/


		dNNInput.data()[j*dimension + 1] = nodeW[0].data()[0];
		dNNInput.data()[j*dimension + 2] = nodeW[0].data()[1];
		dNNInput.data()[j*dimension + 3] = nodeW[0].data()[2];
		dNNInput.data()[j*dimension + 4] = nodeW[0].data()[3];
		dNNInput.data()[j*dimension + 5] = nodePotentialE[j];
		dNNInput.data()[j*dimension + 6] = forceAngle[j];
		dNNInput.data()[j*dimension + 7] = node_distance[j];
	}

}

void RotationTrainingDataGeneratorV3::createInputDatai(VectorXd &lq, VectorXd &dNNInput, VectorXd filedforce, int nodei)
{
	LoboVolumetricMesh::Material* materia = volumtricMesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	double poisson = enmateria->getNu();
	poisson = (poisson - 0.2) / (0.45 - 0.2);

	int dimension = 8;
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(dimension);

	std::vector<Matrix3d> R_list(numVertex);
	std::vector<Vector4d> w_list(numVertex);

	VectorXd w = (*modalRotationSparseMatrix)*lq;
	//volumtricMesh->setDisplacement(lq.data());

	updatePotentialOrder(filedforce);

	//VectorXd Estrain;
	//modalrotationMatrix->computeModalStrainMatrix(Estrain);
	for (int j = 0; j < numVertex; j++)
	{
		//w_list[j] = convertRtoAxisAngle(R_list[j]);
		//Matrix3d R_j;
		//volumtricMesh->computeNodeRotationRing(j, R_j);

		Vector3d w_i;
		for (int k = 0; k < 3; k++)
			w_i.data()[k] = w.data()[j * 3 + k];
		double angle = w_i.norm();
		if (std::abs(angle) > 1e-15)
			w_i.normalize();

		for (int k = 0; k < 3; k++)
			w_list[j][k] = w_i[k];
		w_list[j][3] = angle;
		//w_list[j] = convertRtoAxisAngle(R_j);
	}

		int j = nodei;
		Vector3d nodelq, nodedq;

		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		std::vector<Vector4d> nodeW(1);
		nodeW[0] = w_list[j];

		Matrix3d alignR;
		Vector3d waxis;
		waxis.data()[0] = nodeW[0].data()[0];
		waxis.data()[1] = nodeW[0].data()[1];
		waxis.data()[2] = nodeW[0].data()[2];

		computeNodeAlignRotation(nodelq, waxis, alignR);
		/*std::cout << "node lq" << std::endl;
		std::cout << nodelq.transpose() << std::endl;
		std::cout << "w" << std::endl;
		std::cout << waxis.transpose() << std::endl;*/

		alignRList[j] = alignR;

		AngleAxisd node_aa;
		node_aa = alignR;
		Vector4d node_alignAxisAngle;

		node_alignAxisAngle.data()[0] = node_aa.axis().data()[0];
		node_alignAxisAngle.data()[1] = node_aa.axis().data()[1];
		node_alignAxisAngle.data()[2] = node_aa.axis().data()[2];
		node_alignAxisAngle.data()[3] = node_aa.angle();


		nodelq = alignR*nodelq;
		for (int k = 0; k < nodeW.size(); k++)
		{
			Vector3d temp;
			for (int l = 0; l < 3; l++)
				temp.data()[l] = nodeW[k].data()[l];
			temp = alignR*temp;
			temp.data()[0] = poisson;
			for (int l = 0; l < 3; l++)
				nodeW[k].data()[l] = temp.data()[l];
		}

		dNNInput.data()[0] = nodelq.data()[1];

		dNNInput.data()[1] = nodeW[0].data()[0];
		dNNInput.data()[2] = nodeW[0].data()[1];
		dNNInput.data()[3] = nodeW[0].data()[2];
		dNNInput.data()[4] = nodeW[0].data()[3];
		dNNInput.data()[5] = nodePotentialE[j];
		dNNInput.data()[6] = forceAngle[j];
		dNNInput.data()[7] = node_distance[j];
	
}

void RotationTrainingDataGeneratorV3::convertOutput(Vector3d &output, int nodeid)
{
	output = alignRList[nodeid].transpose()*output;
}

void RotationTrainingDataGeneratorV3::invConvertOutput(Vector3d &output, int nodeid)
{
	output = alignRList[nodeid]*output;
}

void RotationTrainingDataGeneratorV3::testDataByDNNUserDefine(LoboNeuralNetwork* loboNeuralNetwork)
{
	std::cout << "generate data start" << std::endl;
	int totaltrainingSize = trainingModal->getNumTrainingHighFreq() + trainingModal->getNumTrainingSet();
	std::cout << "total training size" << totaltrainingSize << std::endl;
	std::cout << "will export type node: " << this->getExportNodetype() << std::endl;

	std::vector<double> origin_data;
	std::vector<double> target_data;

	std::vector<double> test_origin_data;
	std::vector<double> test_target_data;

	for (int i = 0; i < 100; i++)
	{
		VectorXd inputV(8);
		inputV.setZero();

		inputV.data()[0] = 1;
		inputV.data()[1] = 0.01*i;
		inputV.data()[4] = 1;
		inputV.data()[5] = 0;
		inputV.data()[6] = 0;
		inputV.data()[7] = inputV.data()[5] * inputV.data()[5];

		for (int j = 0; j < 8; j++)
		{
			origin_data.push_back(inputV[j]);
		}

		Vector3d nodedq = loboNeuralNetwork->predictV2(inputV);
		for (int j = 0; j < 3; j++)
		{
			target_data.push_back(nodedq[j]);
		}


		inputV.setZero();

		inputV.data()[0] = 1;
		inputV.data()[1] = 0.01*i;
		inputV.data()[4] = 1;
		inputV.data()[5] = 0;
		inputV.data()[6] = 1;
		inputV.data()[7] = inputV.data()[5] * inputV.data()[5];

		for (int j = 0; j < 8; j++)
		{
			origin_data.push_back(inputV[j]);
		}

		nodedq = loboNeuralNetwork->predictV2(inputV);
		for (int j = 0; j < 3; j++)
		{
			target_data.push_back(nodedq[j]);
		}
	}

	int dimension = 4 + 2 + 2;
	int outputdimension = 3;
	std::cout << "export data" << std::endl;

	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "origin_train_" << exportNodetype << ".txt";
	elefilename[0] = data[0].str();

	data[1] << "target_train_" << exportNodetype << ".txt";
	elefilename[1] = data[1].str();

	data[2] << "origin_test_" << exportNodetype << ".txt";
	elefilename[2] = data[2].str();

	data[3] << "target_test_" << exportNodetype << ".txt";
	elefilename[3] = data[3].str();

	trainingModal->exportExampleData(elefilename[0].c_str(), origin_data.data(), dimension, origin_data.size());
	trainingModal->exportExampleData(elefilename[1].c_str(), target_data.data(), outputdimension, target_data.size());

	trainingModal->exportExampleData(elefilename[2].c_str(), test_origin_data.data(), dimension, test_origin_data.size());
	trainingModal->exportExampleData(elefilename[3].c_str(), test_target_data.data(), outputdimension, test_target_data.size());

	std::cout << "finished export" << std::endl;
}

void RotationTrainingDataGeneratorV3::updatePotentialOrder(VectorXd &exforce, int idnex)
{
	int numVertex = volumtricMesh->getNumVertices();
	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodeforce;
		nodeforce.data()[0] = exforce.data()[j * 3 + 0];
		nodeforce.data()[1] = exforce.data()[j * 3 + 1];
		nodeforce.data()[2] = exforce.data()[j * 3 + 2];
		nodeforce.normalize();
		nodeforce = volumtricMesh->getNodeRestPosition(j);

		nodeforce.data()[0] = 0;
		nodeforce.normalize();
		nodePotentialE[j] = distanceToZeroPotential[j].dot(nodeforce);
	}

	double min = nodePotentialE.minCoeff();
	double max = nodePotentialE.maxCoeff();
	double scale = max - min;

	for (int j = 0; j < numVertex; j++)
	{
		nodePotentialE[j] -= min;
	}
	nodePotentialE /= scale;
}
