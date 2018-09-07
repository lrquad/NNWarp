#pragma once
#include <Eigen/Sparse>
using namespace Eigen;

class ModalRotationMatrix;
class WarpingMapTrainingModel;
class LoboVolumetricMesh;
class TeMeshVoxlizeModel;
class CubeVolumtricMesh;
class LoboVolumetricMeshGraph;
class LoboNeuralNetwork;
class TrainingDataGenerator
{
public:
	TrainingDataGenerator(TeMeshVoxlizeModel* tetmeshvox_,CubeVolumtricMesh* cubemesh_ , LoboVolumetricMesh* volumtricMesh_, WarpingMapTrainingModel* trainingModal_, ModalRotationMatrix* modalrotaionMatrix_, SparseMatrix<double>* modalRotationSparseMatrix_, int numConstrainedDOFs /*= 0*/, int *constrainedDOFs /*= NULL*/);
	~TrainingDataGenerator();

	virtual void generateData();
	virtual void testDataByDNN(LoboNeuralNetwork* loboNeuralNetwork);
	virtual void createInputData(VectorXd &lq,VectorXd &dNNInput);
	virtual void computeNodeDistance(LoboVolumetricMeshGraph* mesh_graph);

	int getExportNodetype() const { return exportNodetype; }
	void setExportNodetype(int val) { exportNodetype = val; }
	bool getSeperateGenerate() const { return seperateGenerate; }
	void setSeperateGenerate(bool val) { seperateGenerate = val; }
	Eigen::Vector3d getZeroPotential() const { return zeroPotential; }
	void setZeroPotential(Eigen::Vector3d val) { zeroPotential = val; }
	Eigen::VectorXd getNodeDistance();

	int getInputDimension() const { return inputDimension; }

	void setInputDimension(int val) { inputDimension = val; }
	int getOutputDimension() const { return outputDimension; }
	void setOutputDimension(int val) { outputDimension = val; }
	int getTestDimension() const { return testDimension; }
	void setTestDimension(int val) { testDimension = val; }
protected:

	bool seperateGenerate;

	ModalRotationMatrix* modalrotationMatrix;
	SparseMatrix<double>* modalRotationSparseMatrix;
	WarpingMapTrainingModel* trainingModal;
	LoboVolumetricMesh* volumtricMesh;
	CubeVolumtricMesh* cubeMesh;
	TeMeshVoxlizeModel* tetVox;

	//store the distance between the node to the nearest boundary.
	std::vector<double> node_distance;
	std::vector<Vector3d> distanceToZeroPotential;
	Vector3d zeroPotential;

	int exportNodetype;

	int numConstrainedDOFs;
	int *constrainedDOFs;

	int inputDimension;
	int outputDimension;
	int testDimension;

};

