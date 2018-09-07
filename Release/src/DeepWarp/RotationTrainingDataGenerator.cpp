#include "RotationTrainingDataGenerator.h"
#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"

RotationTrainingDataGenerator::RotationTrainingDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/) :TrainingDataGenerator(tetmeshvox_, cubemesh_, volumtricMesh_, trainingModal_, modalrotaionMatrix_, modalRotationSparseMatrix_, numConstrainedDOFs, constrainedDOFs)
{
	inputtype = 0;
}

RotationTrainingDataGenerator::~RotationTrainingDataGenerator()
{

}

void RotationTrainingDataGenerator::generateData()
{
	std::cout << "generate data start" << std::endl;
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

	std::vector<double> nodesclae(numVertex);

	for (int i = 0; i < totaltrainingSize; i++)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(i);

		for (int j = 0; j < numVertex; j++)
		{
			/*lq.data()[j * 3 + 2] = 0;
			nq.data()[j * 3 + 2] = 0;*/

			Vector3d nodelq;
			for (int k = 0; k < 3; k++)
			{
				nodelq.data()[k] = lq.data()[j * 3 + k];
			}
			double scale = nodelq.norm();

			if (scale < 1e-10)
			{
				scale = 1e-10;
			}
			scale = 1;
			nodesclae[j] = scale;
		}

		VectorXd dq = nq - lq;

		std::vector<Matrix3d> R_list(numVertex);
		volumtricMesh->setDisplacement(lq.data());
		for (int j = 0; j < numVertex; j++)
		{
			volumtricMesh->computeNodeRotationRing(j, R_list[j]);
		}

		std::vector<Vector4d> w_list(numVertex);

		for (int j = 0; j < numVertex; j++)
		{
			w_list[j] = convertRtoAxisAngle(R_list[j]);
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
					origin_data.push_back(lq.data()[j * 3 + k] / nodesclae[j]);
					target_data.push_back(dq.data()[j * 3 + k] / nodesclae[j]);
				}

				if (inputtype == 0)
				{
					for (int k = 0; k < 9; k++)
					{
						origin_data.push_back(R_list[j].data()[k]);
					}
				}
				else
				{
					for (int k = 0; k < 4; k++)
					{
						origin_data.push_back(w_list[j].data()[k]);
					}
				}

			}
			else
			{
				for (int k = 0; k < 3; k++)
				{
					test_origin_data.push_back(lq.data()[j * 3 + k] / nodesclae[j]);
					test_target_data.push_back(dq.data()[j * 3 + k] / nodesclae[j]);
				}


				if (inputtype == 0)
				{
					for (int k = 0; k < 9; k++)
					{
						test_origin_data.push_back(R_list[j].data()[k]);
					}
				}
				else
				{
					for (int k = 0; k < 4; k++)
					{
						test_origin_data.push_back(w_list[j].data()[k]);
					}
				}
			}

			for (int n = 0; n < 6; n++)
			{
				Vector4d w;
				w.setZero();
				Matrix3d R;
				R.setZero();

				int neighborid = element->neighbors[n];
				if (neighborid != -1)
				{
					int cubesize = tetVox->getCubeNodeSize(neighborid);
					if (cubesize!=0)
					for (int k = 0; k < 1; k++)
					{
						int nodeindex = tetVox->getCubeNode(neighborid, k);
						w = w_list[nodeindex];
						R = R_list[nodeindex];
					}
				}

				if (i % 5 != 0)
				{
					if (inputtype == 0)
					{
						for (int k = 0; k < 9; k++)
							origin_data.push_back(R.data()[k]);
					}
					else
					{
						for (int k = 0; k < 4; k++)
							origin_data.push_back(w.data()[k]);
					}
				}
				else
				{
					if (inputtype == 0)
					{
						for (int k = 0; k < 9; k++)
							test_origin_data.push_back(R.data()[k]);
					}
					else
					{
						for (int k = 0; k < 4; k++)
							test_origin_data.push_back(w.data()[k]);
					}
				}
			}

		}
	}

	int dimension = 0;
	if (inputtype == 0)
	{
		dimension = 66;
	}
	else
	{
		dimension = 31;
	}
	
	std::cout << "export data" << std::endl;

	trainingModal->exportExampleData("origin_train.txt", origin_data.data(), dimension, origin_data.size());
	trainingModal->exportExampleData("target_train.txt", target_data.data(), 3, target_data.size());

	trainingModal->exportExampleData("origin_test.txt", test_origin_data.data(), dimension, test_origin_data.size());
	trainingModal->exportExampleData("target_test.txt", test_target_data.data(), 3, test_target_data.size());

}

void RotationTrainingDataGenerator::createInputData(VectorXd &lq, VectorXd &dNNInput)
{
	int dimension = 0;
	if (inputtype == 0)
	{
		dimension = 66;
	}
	else
	{
		dimension = 31;
	}

	int numVertex = volumtricMesh->getNumVertices();
	dNNInput.resize(numVertex * dimension);

	std::vector<Matrix3d> R_list(numVertex);
	std::vector<Vector4d> w_list(numVertex);

	volumtricMesh->setDisplacement(lq.data());

	for (int j = 0; j < numVertex; j++)
	{
		volumtricMesh->computeNodeRotationRing(j, R_list[j]);
		w_list[j] = convertRtoAxisAngle(R_list[j]);
	}

	std::vector<double> nodesclae(numVertex);


	for (int i = 0; i < numVertex; i++)
	{

		for (int k = 0; k < 3; k++)
		{
			dNNInput.data()[i * dimension + k] = lq.data()[i * 3 + k];
		}

		if (inputtype == 0)
		{
			for (int k = 3; k < 12; k++)
			{
				dNNInput.data()[i * dimension + k] = R_list[i].data()[k - 3];
			}
		}
		else
		{
			for (int k = 3; k < 7; k++)
			{
				dNNInput.data()[i * dimension + k] = w_list[i].data()[k - 3];
			}
		}

		int cubeid = tetVox->getNodeCube(i);
		CubeElement* element = cubeMesh->getCubeElement(cubeid);

		//if (0)
		for (int n = 0; n < 6; n++)
		{
			Matrix3d R;
			R.setZero();

			Vector4d w;
			w.setZero();

			int neighborid = element->neighbors[n];
			if (neighborid != -1)
			{
				int cubesize = tetVox->getCubeNodeSize(neighborid);
				if (cubesize != 0)
					for (int k = 0; k < 1; k++)
					{
						int nodeindex = tetVox->getCubeNode(neighborid, k);
						w = w_list[nodeindex];
						R = R_list[nodeindex];
					}
			}

			if (inputtype == 0)
			{
				for (int k = 0; k < 9; k++)
				{
					dNNInput.data()[i * dimension + 12 + n * 9 + k] = R.data()[k];
				}
			}
			else
			{
				for (int k = 0; k < 4; k++)
				{
					dNNInput.data()[i * dimension + 7 + n * 4 + k] = w.data()[k];
				}
			}
		}
	}

}

Vector4d RotationTrainingDataGenerator::convertRtoAxisAngle(Matrix3d R)
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
