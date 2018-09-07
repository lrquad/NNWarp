#pragma once

#include "RotationTrainingDataGeneratorV2.h"

class LoboNeuralNetwork;

class RotationTrainingDataGeneratorV3 :public RotationTrainingDataGeneratorV2
{
public:
	RotationTrainingDataGeneratorV3(TeMeshVoxlizeModel* tetmeshvox_, CubeVolumtricMesh* cubemesh_, LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/);
	~RotationTrainingDataGeneratorV3();

	virtual void generateData();
	virtual void testDataByDNN(LoboNeuralNetwork* loboNeuralNetwork);
	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput);
	virtual void createInputData(VectorXd &lq, VectorXd &dNNInput, VectorXd filedforce);
	virtual void createInputDatai(VectorXd &lq,VectorXd &dNNinput, VectorXd filedforce,int nodei);

	virtual void convertOutput(Vector3d &output, int nodeid);
	virtual void invConvertOutput(Vector3d &output, int nodeid);

	virtual void updatePotentialOrder(VectorXd &exforce, int idnex = -1);

	bool getForcefieldType() const { return forcefieldType; }
	void setForcefieldType(bool val) { forcefieldType = val; }
	Eigen::VectorXd getNodePotentialE() const { return nodePotentialE; }
	Eigen::VectorXd getNodeForceAngle() const { return forceAngle; }
	void setNodePotentialE(Eigen::VectorXd val) { nodePotentialE = val; }
protected:

	virtual void testDataByDNNUserDefine(LoboNeuralNetwork* loboNeuralNetwork);


	std::vector<Matrix3d> alignRList;
	VectorXd nodePotentialE;
	VectorXd nodeForceCoordinate;
	VectorXd forceAngle;
	VectorXd nodeMass;

	bool forcefieldType;

};

