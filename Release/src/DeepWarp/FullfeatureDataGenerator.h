#pragma once

#include "RotationStrainDataGenerator.h"

class FullfeatureDataGenerator :public RotationStrainDataGenerator
{
public:
	FullfeatureDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/);
	~FullfeatureDataGenerator();

	virtual void generateData();

	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput, VectorXd &force);
};

