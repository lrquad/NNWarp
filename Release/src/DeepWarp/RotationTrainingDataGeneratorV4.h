#pragma once

#include "RotationTrainingDataGeneratorV3.h"

class LoboNeuralNetwork;
class RotateForceField;

class RotationTrainingDataGeneratorV4 :public RotationTrainingDataGeneratorV3
{
public:
	RotationTrainingDataGeneratorV4(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/,RotateForceField* forcefield_);
	~RotationTrainingDataGeneratorV4();
	virtual void updatePotentialOrder(VectorXd &exforce, int idnex = -1);

protected:
	RotateForceField* forcefield;

	virtual void computeSphereCoordinates(Vector3d &v1, Vector3d &v2);

};

