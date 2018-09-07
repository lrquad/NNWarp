#pragma once

#include "RotationTrainingDataGeneratorV3.h"

class RotationStrainDataGenerator :public RotationTrainingDataGeneratorV3
{
public:
	RotationStrainDataGenerator(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/);
	~RotationStrainDataGenerator();

	virtual void generateData();

	virtual void generateDataForPlot();

	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput);
	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput,VectorXd &force);


	//dimension =  oridimension + 9 (rotation matrix)
	virtual void createInputDataWithRotation(VectorXd &lq, VectorXd &dNNInput, VectorXd &force);

	bool getApplyalignRotation() const { return applyalignRotation; }
	void setApplyalignRotation(bool val) { applyalignRotation = val; }

	virtual void updatePotentialOrder(VectorXd &exforce, int idnex = -1);


protected:


	virtual void pushTrainingData(VectorXd &input,VectorXd &output);
	virtual void pushTestData(VectorXd &input,VectorXd &output);


	std::vector<bool> ifnodeconstrained;
	std::vector<double> origin_data;
	std::vector<double> target_data;

	std::vector<double> test_origin_data;
	std::vector<double> test_target_data;

	bool applyalignRotation;
	bool dynamicForceDirection;
};

