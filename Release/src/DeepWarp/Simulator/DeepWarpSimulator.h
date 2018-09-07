#pragma once
#include "Simulator/simulatorBase/LoboSimulatorBase.h"

class InvertibleSTVKModel;
class LinearSTVkForceModel;
class STVKForceModel;
class ModalRotationMatrix;
class LoboNeuralNetwork;
class WarpingMapTrainingModel;
class CubeVolumtricMesh;
class TeMeshVoxlizeModel;
class TrainingDataGenerator;

class IsotropicHyperlasticForceModel;
class LoboIsotropicMaterial;
class IsotropicHyperelasticModel;
class RotateForceField;
class ReducedSTVKModel;
class ReducedForceModel;

class DeepWarpSimulator :public LoboSimulatorBase
{
public:
	DeepWarpSimulator(std::ifstream &readconfig, LoboTriMesh* obj_mesh, bool ifreadconfig);
	~DeepWarpSimulator();

	//preset all pointer to cNULL
	void nullAllPointer();

	//call this function in destructor 
	void deleteAllPointer();

	virtual void initSimulator(int verbose = 0);

	//simulation
	virtual void updateSimulator(int verbose = 0);
	virtual void resetSimulator(int verbose = 0);

	//return current simulator type
	virtual LoboSimulatorBase::simulatorType getType();

	virtual void saveSimulator(const char* filename, fileFormatType formattype) const;
	virtual void readSimulator(const char* filename, fileFormatType formattype);

	virtual void readVolumetricMeshAscii(const char* filenamebase);
	virtual void readCubeMeshAscii(const char* filenamebase);
	virtual void creadCubeMeshDistanceAscii(const char* filenamebase);
	virtual void exportConvexCubeMesh(const char* filenamebase);

	virtual void initSimulatorGlobalConstraints();

	virtual void getNodeOrientation(VectorXd &nodeYaxis);
	virtual void getNodeOrientation(VectorXd &nodeYaxis,Vector3d axis);


	/* =============================
	mouse event
	=============================*/
	virtual void singleClickEvent();
	virtual void rectSelectEvent();
	virtual void mouseMoveEvent();
	virtual void mouseReleaseEvent();

	void initSubspaceModes();
	void readSubspaceModes(const char* filename);
	void saveSubspaceModes(const char* filename);

	//training Data generate
	void initTrainingDataModel();
	void doGenerateTrainingDataModel();
	void doGenerateDataTestedByDNN();

	void searchClosestTrainingData(double threshold);
	void updateVolMeshDis(int disIndex, bool nonlinear = false);

	void saveTrainingData(const char* filename);
	void readTrainingData(const char* filename);

	void getplotTrainingData(std::vector<double> &x, std::vector<double> &y);

	void loadTrainingDataFeature(std::vector<double> &x, std::vector<double> &y,const char* filename);
	void loadTrainingDataPCAFeature(std::vector<double> &x, std::vector<double>&y);


	void loadTrainingDataPCA(const char* filename);

	void getSelectedNodesFeature(std::vector<double> &x, std::vector<double> &y);
	void getSelectedNodesPCAFeature(std::vector<double> &x, std::vector<double> &y);


	VectorXd getMergedExternalForce() const { return mergedExternalForce; }
	void setMergedExternalForce(VectorXd val) { mergedExternalForce = val; }
	WarpingMapTrainingModel* getTrainingModal() const { return trainingModal; }
	void setTrainingModal(WarpingMapTrainingModel* val) { trainingModal = val; }
	CubeVolumtricMesh* getCubeVolumtricMesh() const { return cubeVolumtricMesh; }
	void setCubeVolumtricMesh(CubeVolumtricMesh* val) { cubeVolumtricMesh = val; }

	void getNodePotentialValue();

	//0 distance 1 potential 2 digression
	void exportFeatureMap(int featureid);

	bool getNonlinearMapping() const { return nonlinearMapping; }
	void setNonlinearMapping(bool val) { nonlinearMapping = val; }
	
	int getNodeNeighborTypeExport() const { return nodeNeighborTypeExport; }
	void setNodeNeighborTypeExport(int val) { nodeNeighborTypeExport = val; }
	
	bool getFinishedInit() const { return finishedInit; }
	void setFinishedInit(bool val) { finishedInit = val; }

	bool getModalwarpingMapping() const { return modalwarpingMapping; }
	void setModalwarpingMapping(bool val) { modalwarpingMapping = val; }
	bool getSeperateGenerateData() const { return seperateGenerateData; }
	void setSeperateGenerateData(bool val) { seperateGenerateData = val; }


	virtual void generateLinearRotationTrainingData();

	virtual void generateEnergyCurveData(); 

	void setGenLinearShapeOnly(bool b);

	bool getGPUversion() const { return GPUversion; }
	void setGPUversion(bool val) { GPUversion = val; }
protected:

	bool finishedInit;

	virtual void initIntegrator();
	virtual void readConfig(std::ifstream &instream);
	void mergeExternalForce();

	void exportFeatureData(const char* filename, std::vector<double> features, int dimension);
	void exportTargetData(const char* filename, std::vector<double> targets, int dimension);


	SparseMatrix<double>* massMatrix;
	SparseMatrix<double>* modalRotationSparseMatrix;
	SparseMatrix<double>* rotationSparseMatrixR;
	SparseMatrix<double>* localOrientationMatrixR;
	MatrixXd reduced_modes;
	VectorXd mergedExternalForce;


	//rotation train coordinate method
	SparseMatrix<double>* V;
	//deformation gradient operator
	SparseMatrix<double>* G;
	SparseMatrix<double>* w_ele;
	SparseMatrix<double>* e_ele;
	SparseMatrix<double> rotationStrainMatrix;
	SparseMatrix<double> subRotationStrainMatrix;
	std::vector<int> mapOldNew;
	std::vector<int> mapSparseMatrixEntryOldNew;

	SimplicialLDLT<SparseMatrix<double>> LDLT;


	InvertibleSTVKModel* invertibleStvkModel;
	STVKForceModel* stvkforce_model;
	STVKForceModel* stvkforceNonlinear_model;
	ModalRotationMatrix* modalrotationMatrix;
	LoboNeuralNetwork* loboNeuralNetwork;

	std::vector<LoboNeuralNetwork*> loboNeuralNetworkList;


	WarpingMapTrainingModel* trainingModal;
	CubeVolumtricMesh* cubeVolumtricMesh;
	CubeVolumtricMesh* cubeVolumtricMesh_distace;
	TeMeshVoxlizeModel* tetmeshVoxModel;
	TrainingDataGenerator* dataGenerator;

	IsotropicHyperlasticForceModel* hyperforcemodel;
	LoboIsotropicMaterial* isotropicMaterial;
	IsotropicHyperelasticModel* isotropicHyperelasticModel;
	RotateForceField* forcefield;


	std::string DNNpath;
	bool nonlinearMapping;
	bool modalwarpingMapping;
	bool seperateGenerateData;

	int modesk;
	int modesr;
	int showModeIndex;
	int nodeNeighborTypeExport;


	VectorXd selectedNodeInput;
	int selectedNodeid;
	//para for generate training data
	double domainscale;

	VectorXd cur_total_force;

	//for data analysis
	MatrixXd dataset;
	MatrixXd outputpca;
	MatrixXd transformpca;
	RowVectorXd scalingFactors;
	RowVectorXd minCoeffs;
	RowVectorXd featureMeans;

	std::vector<double> nodetrajectory;
	std::vector<int> singleSelectedNeighborNodes;
	std::vector<double> neighbordistance;

	bool GPUversion;

};

DeepWarpSimulator* downCastModalwarpingSimulator(LoboSimulatorBase* simulator);