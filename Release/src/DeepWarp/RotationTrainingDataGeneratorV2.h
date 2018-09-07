#pragma once
#include "TrainingDataGenerator.h"
class LoboNeuralNetwork;
class RotationTrainingDataGeneratorV2 :public TrainingDataGenerator
{
public:
	RotationTrainingDataGeneratorV2(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/);
	~RotationTrainingDataGeneratorV2();

	virtual void generateData();
	virtual void testDataByDNN(LoboNeuralNetwork* loboNeuralNetwork);

	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput);
	virtual void convertOutput(VectorXd &output);
	virtual void convertOutput(Vector3d &output, int nodeid);

	int getInputtype() const { return inputtype; }
	void setInputtype(int val) { inputtype = val; }

protected:

	virtual void generateDataSub();

	virtual Vector4d convertRtoAxisAngle(Matrix3d R);

	virtual void computeNodeAlignRotation(Vector3d nodelq, Vector3d axis, Matrix3d &rotation);

	SparseMatrix<double>* rotationSparseMatrixR;
	int inputtype;

};

