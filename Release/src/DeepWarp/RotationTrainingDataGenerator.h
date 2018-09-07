#pragma once
#include "TrainingDataGenerator.h"


class RotationTrainingDataGenerator:public TrainingDataGenerator
{
public:
	RotationTrainingDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/);
	~RotationTrainingDataGenerator();

	virtual void generateData();
	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput);


	int getInputtype() const { return inputtype; }
	void setInputtype(int val) { inputtype = val; }
protected:

	virtual Vector4d convertRtoAxisAngle(Matrix3d R);

	int inputtype; //0 rotation 1 axis angle
};

