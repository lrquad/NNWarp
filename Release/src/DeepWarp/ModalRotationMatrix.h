#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

class LoboVolumetricMesh;
class TetVolumetricMesh;

class ModalRotationMatrix
{
public:
	ModalRotationMatrix(TetVolumetricMesh* volumtricMesh_);
	~ModalRotationMatrix();

	virtual void computeModalRotationMatrix_W(MatrixXd& W);
	
	virtual void computeModalRotationSparseMatrix_W(SparseMatrix<double>* W);
	virtual void computeModalRotationSparseMatrix_E(SparseMatrix<double>* E);


	virtual void computeModalRotationSparseMatrix_w_perele(SparseMatrix<double>* W);
	virtual void computeModalRotationSparseMatrix_E_perele(SparseMatrix<double>* E);
	
	//w is 3m X 1  e is 6m X 1 and g is 9m X 1
	virtual void computeRotationStran_To_g(VectorXd &w, VectorXd &e, VectorXd &g);
	
	virtual void computeRotationStrain_To_g_node(VectorXd &w, VectorXd &e, VectorXd &g);


	virtual void computeModalStrainMatrix(VectorXd &E,double mean = 0,double max = 1, double min =0);

	//R 
	virtual void computeWarpingRotationMatrixR(SparseMatrix<double>* R_,VectorXd w);
	virtual void computeWarpingRotationmatrixR_subnodes(SparseMatrix<double>* R_, VectorXd w,std::vector<int> &nodelist);

	//get rotation matrix by vertex w
	virtual void computeWarpingRotationMatrixRi(VectorXd &w, int vertexid, Matrix3d & rotationR);
	
	virtual void computeLocalOrientationMatrixR(SparseMatrix<double>* R_, VectorXd w,bool transpose = true);

	virtual void computeLocalOrientationByPolarDecomposition(SparseMatrix<double>* R_, VectorXd q, bool transpose = true);

	SparseMatrix<double>* w_operator;
	SparseMatrix<double>* e_operator;
	SparseMatrix<double>* g_operator;

	std::vector<double> nodeTotalVolume; // total element volume

protected:

	TetVolumetricMesh* volumtricMesh;
	std::vector<Matrix3d> nodeStrain;




};

