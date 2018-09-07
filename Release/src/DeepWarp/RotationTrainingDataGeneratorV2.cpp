#include "RotationTrainingDataGeneratorV2.h"
#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"
#include "Functions/GeoMatrix.h"
#include <fstream>
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"


RotationTrainingDataGeneratorV2::RotationTrainingDataGeneratorV2(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/) :TrainingDataGenerator(tetmeshvox_, cubemesh_, volumtricMesh_, trainingModal_, modalrotaionMatrix_, modalRotationSparseMatrix_, numConstrainedDOFs, constrainedDOFs)
{
	inputtype = 0;
	rotationSparseMatrixR = new SparseMatrix<double>();
}

RotationTrainingDataGeneratorV2::~RotationTrainingDataGeneratorV2()
{
	delete rotationSparseMatrixR;
}

void RotationTrainingDataGeneratorV2::generateData()
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

	for (int i = 0; i < totaltrainingSize; i+=2)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd w = (*modalRotationSparseMatrix)*lq;
		double nqNorm = nq.norm();

		/*modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);
		nq = (*rotationSparseMatrixR)*lq;*/

		VectorXd dq = nq;
		//trainingModal->setTrainingNonLinearDis(dq, i);

		volumtricMesh->setDisplacement(lq.data());

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

		for (int j = 0; j < numVertex; j++)
		{
			if (ifnodeconstrained[j])
			{
				continue;
			}

			int cubeid = tetVox->getNodeCube(j);
			CubeElement* element = cubeMesh->getCubeElement(cubeid);
			CubeNodeMapType* cubenodemap = tetVox->getCubeNodeMapRefer(j);

			if (getSeperateGenerate())
			if (cubenodemap->getMainType() != this->getExportNodetype())
			{
				continue;
			}

			Vector3d nodelq, nodedq;

			for (int k = 0; k < 3; k++)
			{
				nodelq.data()[k] = lq.data()[j * 3 + k];
				nodedq.data()[k] = dq.data()[j * 3 + k];
			}

			std::vector<Vector4d> nodeW(7);
			nodeW[0] = w_list[j];

			for (int k = 0; k < 6; k++)
			{
				int neighborid = element->neighbors[k];
				
				nodeW[k + 1].setZero();

				if (neighborid != -1)
				{
					int cubesize = tetVox->getCubeNodeSize(neighborid);
					if (cubesize != 0)
					{
						int nodeindex = tetVox->getCubeNode(neighborid, 0);
						int orderedindex = cubenodemap->reorderNeighbor[k];
						nodeW[orderedindex + 1] = w_list[nodeindex];
						nodeW[k + 1] = w_list[nodeindex];
					}
				}
			}

			//apply local rotation
			if (0)
			{
				nodelq = cubenodemap->R_*nodelq;
				nodedq = cubenodemap->R_*nodedq;
				//nodeTransFormation = cubenodemap->R_*nodeTransFormation;

				for (int k = 0; k < 7; k++)
				{
					Vector3d temp;
					for (int l = 0; l < 3; l++)
						temp.data()[l] = nodeW[k].data()[l];

					temp = cubenodemap->R_*temp;

					for (int l = 0; l < 3; l++)
						nodeW[k].data()[l] = temp.data()[l];
				}
			}


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

				for (int k = 0; k < 7; k++)
				{
					Vector3d temp;
					for (int l = 0; l < 3; l++)
						temp.data()[l] = nodeW[k].data()[l];

					temp = alignR*temp;

					for (int l = 0; l < 3; l++)
						nodeW[k].data()[l] = temp.data()[l];
				}
			}

			int scaleNum = 1;
			Vector3d originlq = nodelq;
			Vector3d origindq = nodedq;
			while (scaleNum)
			{
				nodelq = originlq* 1.0 / scaleNum;
				nodedq = origindq* 1.0 / scaleNum;

				if (i % 5 != 0)
				{
					//only need y 
					origin_data.push_back(node_distance[j]);
					origin_data.push_back(nodelq.data()[1]);

					for (int k = 0; k < 3; k++)
					{
						target_data.push_back(nodedq.data()[k]);
					}

					for (int k = 0; k < 9; k++)
					{
						//target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < 7; k++)
					{
						for (int l = 0; l < 4; l++)
						{
							origin_data.push_back(nodeW[k].data()[l]);
						}
					}

					for (int k = 0; k < 4; k++)
					{
						//origin_data.push_back(node_alignAxisAngle.data()[k]);
					}

					data_weight.push_back(nqNorm);

				}
				else
				{
					test_origin_data.push_back(node_distance[j]);
					test_origin_data.push_back(nodelq.data()[1]);

					for (int k = 0; k < 3; k++)
					{
						test_target_data.push_back(nodedq.data()[k]);
					}

					for (int k = 0; k < 9; k++)
					{
						//test_target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < 7; k++)
					{
						for (int l = 0; l < 4; l++)
						{
							test_origin_data.push_back(nodeW[k].data()[l]);
						}
					}

					for (int k = 0; k < 4; k++)
					{
						//test_origin_data.push_back(node_alignAxisAngle.data()[k]);
					}
				}
				scaleNum--;
			}
		}

	}

	int dimension = 30;
	int outputdimension = 3;
	std::cout << "export data" << std::endl;

	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "origin_train_" << exportNodetype<< ".txt";
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

void RotationTrainingDataGeneratorV2::testDataByDNN(LoboNeuralNetwork* loboNeuralNetwork)
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

	//less data to show
	int datacount = 0;
	for (int i = 0; i < totaltrainingSize; i+=1)
	{

		if (i <= totaltrainingSize-50)
		{
			continue;
		}

		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd w = (*modalRotationSparseMatrix)*lq;


		/*modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);
		nq = (*rotationSparseMatrixR)*lq;*/

		VectorXd dq = nq;
		//trainingModal->setTrainingNonLinearDis(dq, i);

		volumtricMesh->setDisplacement(lq.data());


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

		for (int j = 0; j < numVertex; j++)
		{
			if (ifnodeconstrained[j])
			{
				continue;
			}

			int cubeid = tetVox->getNodeCube(j);
			CubeElement* element = cubeMesh->getCubeElement(cubeid);
			CubeNodeMapType* cubenodemap = tetVox->getCubeNodeMapRefer(j);

			if (getSeperateGenerate())
			if (cubenodemap->getMainType() != this->getExportNodetype())
			{
				continue;
			}

			Vector3d nodelq, nodedq;

			for (int k = 0; k < 3; k++)
			{
				nodelq.data()[k] = lq.data()[j * 3 + k];
				nodedq.data()[k] = dq.data()[j * 3 + k];
			}

			std::vector<Vector4d> nodeW(7);
			nodeW[0] = w_list[j];

			for (int k = 0; k < 6; k++)
			{
				int neighborid = element->neighbors[k];

				nodeW[k + 1].setZero();

				if (neighborid != -1)
				{
					int cubesize = tetVox->getCubeNodeSize(neighborid);
					if (cubesize != 0)
					{
						int nodeindex = tetVox->getCubeNode(neighborid, 0);
						int orderedindex = cubenodemap->reorderNeighbor[k];
						nodeW[orderedindex + 1] = w_list[nodeindex];
						nodeW[k + 1] = w_list[nodeindex];
					}
				}
			}

			//apply local rotation
			if (0)
			{
				nodelq = cubenodemap->R_*nodelq;
				nodedq = cubenodemap->R_*nodedq;
				//nodeTransFormation = cubenodemap->R_*nodeTransFormation;

				for (int k = 0; k < 7; k++)
				{
					Vector3d temp;
					for (int l = 0; l < 3; l++)
						temp.data()[l] = nodeW[k].data()[l];

					temp = cubenodemap->R_*temp;

					for (int l = 0; l < 3; l++)
						nodeW[k].data()[l] = temp.data()[l];
				}
			}


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

				for (int k = 0; k < 7; k++)
				{
					Vector3d temp;
					for (int l = 0; l < 3; l++)
						temp.data()[l] = nodeW[k].data()[l];

					temp = alignR*temp;

					for (int l = 0; l < 3; l++)
						nodeW[k].data()[l] = temp.data()[l];
				}
			}

			int scaleNum = 1;
			Vector3d originlq = nodelq;
			Vector3d origindq = nodedq;
			while (scaleNum)
			{
				nodelq = originlq* 1.0 / scaleNum;
				nodedq = origindq* 1.0 / scaleNum;

				if (datacount % 5 != 0)
				{
					//only need y 
					origin_data.push_back(node_distance[j]);
					origin_data.push_back(nodelq.data()[1]);
					for (int k = 0; k < 9; k++)
					{
						//target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < 7; k++)
					{
						for (int l = 0; l < 4; l++)
						{
							origin_data.push_back(nodeW[k].data()[l]);
						}
					}

					VectorXd inputv(30);
					for (int i = 0; i < 30; i++)
					{
						inputv[i] = origin_data.data()[origin_data.size() - 30 + i];
					}

					nodedq = loboNeuralNetwork->predictV2(inputv);

					for (int k = 0; k < 3; k++)
					{
						target_data.push_back(nodedq.data()[k]);
					}


					for (int k = 0; k < 4; k++)
					{
						//origin_data.push_back(node_alignAxisAngle.data()[k]);
					}

				}
				else
				{
					test_origin_data.push_back(node_distance[j]);
					test_origin_data.push_back(nodelq.data()[1]);
					for (int k = 0; k < 9; k++)
					{
						//test_target_data.push_back(nodeTransFormation.data()[k]);
					}

					for (int k = 0; k < 7; k++)
					{
						for (int l = 0; l < 4; l++)
						{
							test_origin_data.push_back(nodeW[k].data()[l]);
						}
					}

					VectorXd inputv(30);
					for (int i = 0; i < 30; i++)
					{
						inputv[i] = test_origin_data.data()[test_origin_data.size() - 30 + i];
					}

					//nodedq = loboNeuralNetwork->predictV2(inputv);

					for (int k = 0; k < 3; k++)
					{
						test_target_data.push_back(nodedq.data()[k]);
					}

					for (int k = 0; k < 4; k++)
					{
						//test_origin_data.push_back(node_alignAxisAngle.data()[k]);
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
		trainingModal->setTrainingNonLinearDis(dq, i);
		datacount++;
	}

	int dimension = 30;
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

void RotationTrainingDataGeneratorV2::createInputData(VectorXd &lq, VectorXd &dNNInput)
{
	int dimension = 30;
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);

	std::vector<Matrix3d> R_list(numVertex);
	std::vector<Vector4d> w_list(numVertex);

	VectorXd w = (*modalRotationSparseMatrix)*lq;
	volumtricMesh->setDisplacement(lq.data());


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
		int cubeid = tetVox->getNodeCube(j);
		CubeElement* element = cubeMesh->getCubeElement(cubeid);
		CubeNodeMapType* cubenodemap = tetVox->getCubeNodeMapRefer(j);

		Vector3d nodelq, nodedq;

		for (int k = 0; k < 3; k++)
		{
			nodelq.data()[k] = lq.data()[j * 3 + k];
		}

		std::vector<Vector4d> nodeW(7);
		nodeW[0] = w_list[j];
		for (int k = 0; k < 6; k++)
		{
			int neighborid = element->neighbors[k];
			nodeW[k + 1].setZero();
			if (neighborid != -1)
			{
				int cubesize = tetVox->getCubeNodeSize(neighborid);
				if (cubesize != 0)
				{
					int nodeindex = tetVox->getCubeNode(neighborid, 0);
					int orderedindex = cubenodemap->reorderNeighbor[k];
					nodeW[orderedindex + 1] = w_list[nodeindex];
					nodeW[k + 1] = w_list[nodeindex];
				}
			}
		}

		if (0)
		{
			nodelq = cubenodemap->R_*nodelq;

			for (int k = 0; k < 7; k++)
			{
				Vector3d temp;
				for (int l = 0; l < 3; l++)
					temp.data()[l] = nodeW[k].data()[l];
				temp = cubenodemap->R_*temp;
				for (int l = 0; l < 3; l++)
					nodeW[k].data()[l] = temp.data()[l];
			}
		}


		Matrix3d alignR;
		Vector3d waxis;
		waxis.data()[0] = nodeW[0].data()[0];
		waxis.data()[1] = nodeW[0].data()[1];
		waxis.data()[2] = nodeW[0].data()[2];

		/*computeRotationBetween2Vec(nodelq, Vector3d(0, -1, 0), alignR);
		std::cout << (alignR*nodedq).transpose() << std::endl;
		std::cout << (alignR*waxis).transpose() << std::endl;
		std::cout << (alignR*nodelq).transpose() << std::endl;*/

		computeNodeAlignRotation(nodelq, waxis, alignR);
		
		cubenodemap->R_align_ = alignR;

		

		AngleAxisd node_aa;
		node_aa = alignR;
		Vector4d node_alignAxisAngle;
		
		node_alignAxisAngle.data()[0] = node_aa.axis().data()[0];
		node_alignAxisAngle.data()[1] = node_aa.axis().data()[1];
		node_alignAxisAngle.data()[2] = node_aa.axis().data()[2];
		node_alignAxisAngle.data()[3] = node_aa.angle();


		nodelq = alignR*nodelq;
		for (int k = 0; k < 7; k++)
		{
			Vector3d temp;
			for (int l = 0; l < 3; l++)
				temp.data()[l] = nodeW[k].data()[l];
			temp = alignR*temp;
			for (int l = 0; l < 3; l++)
				nodeW[k].data()[l] = temp.data()[l];
		}

		dNNInput.data()[j*dimension + 0] = node_distance[j];
		dNNInput.data()[j*dimension + 1] = nodelq.data()[1];



		for (int k = 0; k < 7; k++)
		{
			for (int l = 0; l < 4; l++)
			{
				dNNInput.data()[j*dimension + 2 + k * 4 + l] = nodeW[k].data()[l];
			}
		}
	}
}

void RotationTrainingDataGeneratorV2::convertOutput(VectorXd &output)
{
	int numVertex = volumtricMesh->getNumVertices();
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d temp;
		CubeNodeMapType* cubenodemap = tetVox->getCubeNodeMapRefer(i);
		for (int j = 0; j < 3; j++)
		{
			temp.data()[j] = output.data()[i * 3 + j];
		}

		temp = cubenodemap->R_align_.transpose()*temp;

		for (int j = 0; j < 3; j++)
			output.data()[i * 3 + j];
	}
}

void RotationTrainingDataGeneratorV2::convertOutput(Vector3d &output, int nodeid)
{
	CubeNodeMapType* cubenodemap = tetVox->getCubeNodeMapRefer(nodeid);
	output = cubenodemap->R_align_.transpose()*output;
}

void RotationTrainingDataGeneratorV2::generateDataSub()
{
	
}

Eigen::Vector4d RotationTrainingDataGeneratorV2::convertRtoAxisAngle(Matrix3d R)
{
	AngleAxisd aa;
	aa = R;
	Vector3d axis = aa.axis();
	double angle = aa.angle();

	Vector4d output;
	for (int i = 0; i < 3; i++)
	{
		output.data()[i] = axis.data()[i];
	}

	output.data()[3] = angle;

	return output;
}

void RotationTrainingDataGeneratorV2::computeNodeAlignRotation(Vector3d nodelq, Vector3d axis, Matrix3d &rotation)
{
	Matrix3d alignR;
	computeRotationBetween2Vec(nodelq, Vector3d(0, -1, 0), alignR);
	Vector3d waxis = axis;
	waxis = waxis - (waxis.dot(nodelq.normalized()))*nodelq.normalized();
	waxis = alignR*waxis;
	Matrix3d secondAlignR;
	/*Vector3d secondAxis = Vector3d(0, -1, 0);
	double dotab =
	double angle = std::acos(waxis.dot(Vector3d(0, 0, 1)) / waxis.norm());*/
	computeRotationBetween2Vec(waxis, Vector3d(0, 0, 1), secondAlignR);

	alignR = secondAlignR*alignR;
	rotation = alignR;

	if (nodelq.norm() < 1e-14)
	{
		rotation.setIdentity();
	}
}
