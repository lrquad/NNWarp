#include "TrainingDataGenerator.h"
#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"

TrainingDataGenerator::TrainingDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs_ /*= 0*/, int *constrainedDOFs_ /*= NULL*/)
{
	seperateGenerate = false;
	exportNodetype = 0;
	this->volumtricMesh = volumtricMesh_;
	this->trainingModal = trainingModal_;
	this->modalrotationMatrix = modalrotaionMatrix_;
	this->modalRotationSparseMatrix = modalRotationSparseMatrix_;

	this->numConstrainedDOFs = numConstrainedDOFs_;
	this->constrainedDOFs = constrainedDOFs_;
	this->tetVox = tetmeshvox_;
	this->cubeMesh = cubemesh_;
}

TrainingDataGenerator::~TrainingDataGenerator()
{

}

void TrainingDataGenerator::generateData()
{
	int totaltrainingSize = trainingModal->getNumTrainingHighFreq() + trainingModal->getNumTrainingSet();
	std::cout << "total training size" << totaltrainingSize << std::endl;

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

	for (int i = 0; i < totaltrainingSize; i++)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);
		VectorXd dq = nq - lq;

		VectorXd w = (*modalRotationSparseMatrix)*lq;

		std::vector<Matrix3d> R_list(numVertex);
		for (int j = 0; j < numVertex; j++)
		{
			modalrotationMatrix->computeWarpingRotationMatrixRi(w, j, R_list[j]);
		}

		for (int j = 0; j < numVertex; j++)
		{
			if (ifnodeconstrained[j])
			{
				continue;
			}

			int cubeid = tetVox->getNodeCube(j);
			CubeElement* element = cubeMesh->getCubeElement(cubeid);
			
			if (i % 5 != 0)
			{
				for (int k = 0; k < 3; k++)
				{
					origin_data.push_back(lq.data()[j * 3 + k]);
					target_data.push_back(dq.data()[j * 3 + k]);
				}

				for (int k = 0; k < 9; k++)
				{
					origin_data.push_back(R_list[j].data()[k]);
				}

			}
			else
			{
				for (int k = 0; k < 3; k++)
				{
					test_origin_data.push_back(lq.data()[j * 3 + k]);
					test_target_data.push_back(dq.data()[j * 3 + k]);
				}
				for (int k = 0; k < 9; k++)
				{
					test_origin_data.push_back(R_list[j].data()[k]);
				}
			}
			
			for (int n = 0; n < 6; n++)
			{
				Vector3d ele_w = Vector3d::Zero();

				int neighborid = element->neighbors[n];
				if (neighborid != -1)
				{
					int cubesize = tetVox->getCubeNodeSize(neighborid);
					for (int k = 0; k < cubesize; k++)
					{
						int nodeindex = tetVox->getCubeNode(neighborid, k);
						ele_w.data()[0] += w.data()[nodeindex * 3 + 0];
						ele_w.data()[1] += w.data()[nodeindex * 3 + 1];
						ele_w.data()[2] += w.data()[nodeindex * 3 + 2];
					}
					ele_w /= cubesize;
				}

				if (i % 5 != 0)
				{
					for (int k = 0; k < 3; k++)
						origin_data.push_back(ele_w.data()[k]);
				}
				else
				{
					for (int k = 0; k < 3; k++)
						test_origin_data.push_back(ele_w.data()[k]);
				}
			}

		}
	}

	trainingModal->exportExampleData("origin_train.txt", origin_data.data(), 30, origin_data.size());
	trainingModal->exportExampleData("target_train.txt", target_data.data(), 3, target_data.size());

	trainingModal->exportExampleData("origin_test.txt", test_origin_data.data(), 30, test_origin_data.size());
	trainingModal->exportExampleData("target_test.txt", test_target_data.data(), 3, test_target_data.size());
}

void TrainingDataGenerator::testDataByDNN(LoboNeuralNetwork* loboNeuralNetwork)
{


}

void TrainingDataGenerator::createInputData(VectorXd &lq, VectorXd &dNNInput)
{
	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex*30);
	VectorXd w = (*modalRotationSparseMatrix)*lq;

	std::vector<Matrix3d> R_list(numVertex);
	for (int j = 0; j < numVertex; j++)
	{
		modalrotationMatrix->computeWarpingRotationMatrixRi(w, j, R_list[j]);
	}

	for (int i = 0; i < numVertex; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			dNNInput.data()[i * 30 + k] = lq.data()[i * 3 + k];
		}

		for (int k = 3; k < 12; k++)
		{
			dNNInput.data()[i * 30 + k] = R_list[i].data()[k - 3];
		}

		int cubeid = tetVox->getNodeCube(i);
		CubeElement* element = cubeMesh->getCubeElement(cubeid);
		
		for (int n = 0; n < 6; n++)
		{
			Vector3d ele_w = Vector3d::Zero();

			int neighborid = element->neighbors[n];
			if (neighborid != -1)
			{
				int cubesize = tetVox->getCubeNodeSize(neighborid);
				for (int k = 0; k < cubesize; k++)
				{
					int nodeindex = tetVox->getCubeNode(neighborid, k);
					ele_w.data()[0] += w.data()[nodeindex * 3 + 0];
					ele_w.data()[1] += w.data()[nodeindex * 3 + 1];
					ele_w.data()[2] += w.data()[nodeindex * 3 + 2];
				}
				ele_w /= cubesize;
			}

			for (int k = 0; k < 3; k++)
			{
				dNNInput.data()[i * 30 + 12 + n * 3 + k] = ele_w.data()[k];
			}
		}
	}
}

void TrainingDataGenerator::computeNodeDistance(LoboVolumetricMeshGraph* mesh_graph)
{
	std::vector<std::vector<double>> constrainnodedistance(numConstrainedDOFs / 3);

	for (int i = 0; i < numConstrainedDOFs / 3; i++)
	{
		int nodeid = constrainedDOFs[i * 3] / 3;
		mesh_graph->compute_dijkstra_shortest_paths(nodeid, constrainnodedistance[i]);
	}

	int numVertex = this->volumtricMesh->getNumVertices();
	double max = -DBL_MAX;
	double min = DBL_MAX;
	node_distance.resize(numVertex);

	for (int i = 0; i < numVertex; i++)
	{
		node_distance[i] = DBL_MAX;
		for (int j = 0; j < numConstrainedDOFs / 3; j++)
		{
			if (constrainnodedistance[j][i] < node_distance[i])
			{
				node_distance[i] = constrainnodedistance[j][i];
			}
		}

		if (node_distance[i] > max)
		{
			max = node_distance[i];
		}

		if (node_distance[i] < min)
		{
			min = node_distance[i];
		}

	}

	//scale
	for (int i = 0; i < numVertex; i++)
	{
		//node_distance[i] -= min;
		node_distance[i] /= max;
		//node_distance[i] = 0;
	}

	if (this->trainingModal != NULL)
		this->trainingModal->setNode_distance(node_distance);

	zeroPotential.setZero();
	for (int i = 0; i < numConstrainedDOFs / 3; i++)
	{
		int nodeid = constrainedDOFs[i * 3] / 3;
		Vector3d nodep = volumtricMesh->getNodeRestPosition(nodeid);
		zeroPotential += nodep;
	}
	zeroPotential /= (numConstrainedDOFs/3);

	distanceToZeroPotential.resize(numVertex);
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d nodep = volumtricMesh->getNodeRestPosition(i);
		distanceToZeroPotential[i] = nodep - zeroPotential;
	}
}

Eigen::VectorXd TrainingDataGenerator::getNodeDistance()
{
	int numVertex = volumtricMesh->getNumVertices();
	VectorXd nodedistance(numVertex);
	for (int i = 0; i < numVertex; i++)
	{
		nodedistance[i] = node_distance[i];
	}
	return nodedistance;
}
