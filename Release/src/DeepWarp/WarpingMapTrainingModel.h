#pragma once
#define PI_ 3.14159265359

#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;
class LoboForceModel;
class ImplicitNewMatkSparseIntegrator;

class WarpingMapTrainingModel
{
public:
	WarpingMapTrainingModel(MatrixXd* subspaceModes_,
		int r,double timestep,SparseMatrix<double>* massMatrix_,
		LoboForceModel* forcemodel_,LoboForceModel* nonlinearforceModel_, int numConstrainedDOFs_, int* constrainedDOFs_, double dampingMassCoef, double dampingStiffnessCoef,bool useStaticSolver = true
		);
	
	~WarpingMapTrainingModel();

	virtual void excute();

	//export and import training data
	virtual void exportExampleData(const char* filename, double* features, int dimension, int totalsize);
	virtual void exportExampleDataAscii(const char* filename, double* features, int dimension, int totalsize);


	VectorXd getTrainingLinearDis(int index);
	virtual void getTraingLinearForce(int index, VectorXd &force);
	virtual void getTraingLinearForce(VectorXd& lq, VectorXd &force){};
	virtual void getTrainingLinearForceDirection(int index, Vector3d &force);


	virtual void getTraingLinearForce(VectorXd& lq, VectorXd &force,double poisson){};


	VectorXd getTrainingNonLinearDis(int index);
	Vector3d getForceFieldDirection(int index);
	double getPoissonPerDis(int index);


	void setTrainingNonLinearDis(VectorXd dis, int index);

	double getForceScale() const { return forceScale; }
	void setForceScale(double val) { forceScale = val; }

	int getTotalSizeOfTrainingDis();
	int getTotalSizeOfTestDis();

	int getNumTrainingSet() const { return numTrainingSet; }
	void setNumTrainingSet(int val) { numTrainingSet = val; }
	int getNumTestSet() const { return numTestSet; }
	void setNumTestSet(int val) { numTestSet = val; }
	int getNumTrainingHighFreq() const { return numTrainingHighFreq; }
	void setNumTrainingHighFreq(int val) { numTrainingHighFreq = val; }
	int getNumTestHighFreq() const { return numTestHighFreq; }
	void setNumTestHighFreq(int val) { numTestHighFreq = val; }

	virtual void saveTrainingSet(const char* filename);
	virtual void readTrainingSet(const char* filename);

	bool getAlreadyReadData() const { return alreadyReadData; }
	void setAlreadyReadData(bool val) { alreadyReadData = val; }
	std::vector<double> getNode_distance() const { return node_distance; }
	void setNode_distance(std::vector<double> val) { node_distance = val; }
	bool getForcefieldType() const { return forcefieldType; }
	void setForcefieldType(bool val) { forcefieldType = val; }
	Eigen::Vector3d getPotentialCenter() const { return potentialCenter; }
	void setPotentialCenter(Eigen::Vector3d val) { potentialCenter = val; }
	bool getLinearDisOnly() const { return linearDisOnly; }
	void setLinearDisOnly(bool val) { linearDisOnly = val; }
protected:

	virtual VectorXd rotationGravity(VectorXd gravity, double angle, Vector3d axis);
	virtual VectorXd createGravity(Vector3d nodeacc);

	SparseMatrix<double>* massMatrix;

	std::vector<double> node_distance;

	LoboForceModel* loboforceModel;
	LoboForceModel* nonlinearforceModel;
	ImplicitNewMatkSparseIntegrator* integrator;
	ImplicitNewMatkSparseIntegrator* nonLinearIntegrator;

	MatrixXd* subspaceModes;

	bool alreadyReadData;
	double forceScale;
	int numTrainingSet;
	int numTrainingHighFreq;
	
	int numTestSet;
	int numTestHighFreq;
	int r;

	std::vector<VectorXd> trainingLinearDis;
	std::vector<VectorXd> trainingNonLinearDis;
	std::vector<Vector3d> forcefieldDirection;
	std::vector<double> poissonPerDis;

	std::vector<VectorXd> testLinearDis;
	std::vector<VectorXd> testNonLinearDis;

	int numConstrainedDOFs;
	int* constrainedDOFs;

	bool forcefieldType;
	Vector3d potentialCenter;

	bool linearDisOnly;

};

