#include "RotationTrainingDataGeneratorV4.h"
#include "ModalRotationMatrix.h"
#include "WarpingMapTrainingModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/DeepWarp/Simulator/TeMeshVoxlizeModel.h"
#include "Functions/GeoMatrix.h"
#include <fstream>
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"
#include "Simulator/ForceField/RotateForceField.h"


RotationTrainingDataGeneratorV4::RotationTrainingDataGeneratorV4(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/, RotateForceField* forcefield_) :RotationTrainingDataGeneratorV3(tetmeshvox_, cubemesh_, volumtricMesh_, trainingModal_, modalrotaionMatrix_, modalRotationSparseMatrix_, numConstrainedDOFs, constrainedDOFs)
{
	this->forcefield = forcefield_;
}

RotationTrainingDataGeneratorV4::~RotationTrainingDataGeneratorV4()
{

}

void RotationTrainingDataGeneratorV4::updatePotentialOrder(VectorXd &exforce, int idnex)
{
	int numVertex = volumtricMesh->getNumVertices();
	//Vector3d extforcedirection;
	//extforcedirection.data()[0] = exforce.data()[0];
	//extforcedirection.data()[1] = exforce.data()[1];
	//extforcedirection.data()[2] = exforce.data()[2];
	//extforcedirection.normalize();


	Vector3d axis = forcefield->centeraxis;
	if (idnex != -1)
	{
		axis = trainingModal->getForceFieldDirection(idnex);
	}

	for (int j = 0; j < numVertex; j++)
	{
		Vector3d nodeforce;

		if (getForcefieldType() == 1)
		{
			//nodeforce = volumtricMesh->getNodeRestPosition(j);
			//nodeforce -= forcefield->centeraxis_position;
			//nodeforce = nodeforce - nodeforce.dot(axis)*axis;

			nodePotentialE[j] = -1;
			forceAngle[j] = -1;
		}
		else if (getForcefieldType() == 0)
		{
			nodeforce.data()[0] = exforce.data()[j * 3 + 0];
			nodeforce.data()[1] = exforce.data()[j * 3 + 1];
			nodeforce.data()[2] = exforce.data()[j * 3 + 2];
			nodeforce.normalize();
			nodePotentialE[j] = distanceToZeroPotential[j].dot(nodeforce);
			forceAngle[j] = nodePotentialE[j] / distanceToZeroPotential[j].norm();
		}
	}

	if (getForcefieldType() == 1)
	{
		return;
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

void RotationTrainingDataGeneratorV4::computeSphereCoordinates(Vector3d &v1, Vector3d &v2)
{
	Vector3d axis = v2.cross(v1);
	axis.normalize();
}
