#include "ModalRotationMatrix.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Functions/SkewMatrix.h"

typedef Eigen::Triplet<double> EIGEN_TRI;


ModalRotationMatrix::ModalRotationMatrix(TetVolumetricMesh* volumtricMesh_)
{
	this->volumtricMesh = volumtricMesh_;
	nodeStrain.resize(volumtricMesh->getNumVertices());
	w_operator = new SparseMatrix<double>();
	e_operator = new SparseMatrix<double>();
	g_operator = new SparseMatrix<double>();

	int numVertex = volumtricMesh->getNumVertices();


}

ModalRotationMatrix::~ModalRotationMatrix()
{
	delete w_operator;
	delete e_operator;
	delete g_operator;
}

void ModalRotationMatrix::computeModalRotationMatrix_W(MatrixXd& W)
{

}

void ModalRotationMatrix::computeModalRotationSparseMatrix_W(SparseMatrix<double>* W)
{
	int R = volumtricMesh->getNumVertices() * 3;
	W->resize(R, R);

	double* dis = volumtricMesh->getDisplacementRef();

	int numElement = volumtricMesh->getNumElements();

	Matrix3d Ds;
	Matrix3d F;
	Matrix3d I = Matrix3d::Identity();
	
	MatrixXd ele_W(3, 12);
	ele_W.setZero();
	Matrix3d skewMatrix_v;

	std::vector<EIGEN_TRI> tri_entry_;

	for (int i = 0; i < numElement; i++)
	{
		TetElement* ele =  volumtricMesh->getTetElement(i);
		//ele->computeElementDeformationshapeMatrix(Ds, dis);
		//F = Ds*ele->Dm_inverse - I;
		for (int j = 0; j < 4; j++)
		{
			Vector3d v = ele->Phi_derivate.row(j);
			skewMatrix(v, skewMatrix_v);
			ele_W.block(0, j * 3, 3, 3) = skewMatrix_v;
		}
		ele_W /= 2.0;

		//assign to node
		

		for (int j = 0; j < ele->node_indices.size(); j++)
		{
			int nodeid = ele->node_indices[j];
			LoboNodeBase* node =  volumtricMesh->getNodeRef(nodeid);
			int eleCounter = node->element_list.size();

			for (int l = 0; l < 3; l++)
			{
				for (int m = 0; m < 12; m++)
				{
					int NID = ele->node_indices[m / 3];
					int row = nodeid * 3 + l;
					int col = NID * 3 + m % 3;
					
					double value = ele_W.data()[m * 3 + l] / eleCounter;

					tri_entry_.push_back(EIGEN_TRI(row, col, value));
				}
			}
		}
	}

	W->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}

void ModalRotationMatrix::computeModalRotationSparseMatrix_E(SparseMatrix<double>* E)
{
	int R = volumtricMesh->getNumVertices() * 3;
	int numElements = volumtricMesh->getNumElements();
	int numVertex = volumtricMesh->getNumVertices();

	E->resize(6*numVertex, 3*numVertex);

	std::vector<EIGEN_TRI> tri_entry_;

	MatrixXd E_i(6, 12);
	E_i.setZero();

	for (int i = 0; i < numElements; i++)
	{
		TetElement* ele = volumtricMesh->getTetElement(i);

		/*E_i.data()[0 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 0];
		E_i.data()[3 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 1];
		E_i.data()[6 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 2];
		E_i.data()[9 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 3];
		*/

		E_i.setZero();

		for (int j = 0; j < 4; j++)
		{
			E_i.data()[(j * 3 + 0) * 6 + 0] = 2 * ele->Phi_derivate.data()[0 * 4 + j];

			E_i.data()[(j * 3 + 0) * 6 + 1] = ele->Phi_derivate.data()[1 * 4 + j];
			E_i.data()[(j * 3 + 1) * 6 + 1] = ele->Phi_derivate.data()[0 * 4 + j];

			E_i.data()[(j * 3 + 0) * 6 + 2] = ele->Phi_derivate.data()[2 * 4 + j];
			E_i.data()[(j * 3 + 2) * 6 + 2] = ele->Phi_derivate.data()[0 * 4 + j];


			E_i.data()[(j * 3 + 1) * 6 + 3] = 2 * ele->Phi_derivate.data()[1 * 4 + j];

			E_i.data()[(j * 3 + 1) * 6 + 4] = ele->Phi_derivate.data()[2 * 4 + j];
			E_i.data()[(j * 3 + 2) * 6 + 4] = ele->Phi_derivate.data()[1 * 4 + j];

			E_i.data()[(j * 3 + 2) * 6 + 5] = 2 * ele->Phi_derivate.data()[2 * 4 + j];
		}
		E_i /= 2.0;

		for (int j = 0; j < ele->node_indices.size(); j++)
		{
			int nodeid = ele->node_indices[j];
			LoboNodeBase* node = volumtricMesh->getNodeRef(nodeid);
			int eleCounter = node->element_list.size();

			for (int l = 0; l < 6; l++)
			{
				for (int m = 0; m < 12; m++)
				{
					int NID = ele->node_indices[m / 3];
					int row = nodeid * 6 + l;
					int col = NID * 3 + m % 3;
					double value = E_i.data()[m * 6 + l] / eleCounter;
					tri_entry_.push_back(EIGEN_TRI(row, col, value));
				}
			}
		}
	}
	E->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}

void ModalRotationMatrix::computeModalRotationSparseMatrix_w_perele(SparseMatrix<double>* W)
{
	int numElements = volumtricMesh->getNumElements();
	int numVertex = volumtricMesh->getNumVertices();

	W->resize(numElements * 3, numVertex * 3);

	MatrixXd ele_W(3, 12);
	ele_W.setZero();
	Matrix3d skewMatrix_v;

	std::vector<EIGEN_TRI> tri_entry_;


	for (int i = 0; i < numElements; i++)
	{
		TetElement* ele = volumtricMesh->getTetElement(i);
		//ele->computeElementDeformationshapeMatrix(Ds, dis);
		//F = Ds*ele->Dm_inverse - I;
		for (int j = 0; j < 4; j++)
		{
			Vector3d v = ele->Phi_derivate.row(j);
			skewMatrix(v, skewMatrix_v);
			ele_W.block(0, j * 3, 3, 3) = skewMatrix_v;
		}
		ele_W /= 2.0;

		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 12; k++)
			{
				int nodeid = ele->node_indices[k / 3];
				int col = nodeid * 3 + k % 3;
				int row = i * 3 + j;

				tri_entry_.push_back(EIGEN_TRI(row, col, ele_W.data()[k * 3 + j]));
			}
		}
	}
	W->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}

void ModalRotationMatrix::computeModalRotationSparseMatrix_E_perele(SparseMatrix<double>* E)
{
	int numElements = volumtricMesh->getNumElements();
	int numVertex = volumtricMesh->getNumVertices();

	E->resize(numElements * 6, numVertex * 3);

	std::vector<EIGEN_TRI> tri_entry_;

	MatrixXd E_i(6, 12);
	E_i.setZero();

	for (int i = 0; i < numElements; i++)
	{
		TetElement* ele = volumtricMesh->getTetElement(i);

		/*E_i.data()[0 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 0];
		E_i.data()[3 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 1];
		E_i.data()[6 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 2];
		E_i.data()[9 * 6 + 0] = ele->Phi_derivate.data()[0 * 4 + 3];
		*/

		E_i.setZero();

		for (int j = 0; j < 4; j++)
		{
			E_i.data()[(j * 3 + 0) * 6 + 0] = 2*ele->Phi_derivate.data()[0 * 4 + j];

			E_i.data()[(j * 3 + 0) * 6 + 1] = ele->Phi_derivate.data()[1 * 4 + j];
			E_i.data()[(j * 3 + 1) * 6 + 1] = ele->Phi_derivate.data()[0 * 4 + j];

			E_i.data()[(j * 3 + 0) * 6 + 2] = ele->Phi_derivate.data()[2 * 4 + j];
			E_i.data()[(j * 3 + 2) * 6 + 2] = ele->Phi_derivate.data()[0 * 4 + j];


			E_i.data()[(j * 3 + 1) * 6 + 3] = 2*ele->Phi_derivate.data()[1 * 4 + j];

			E_i.data()[(j * 3 + 1) * 6 + 4] = ele->Phi_derivate.data()[2 * 4 + j];
			E_i.data()[(j * 3 + 2) * 6 + 4] = ele->Phi_derivate.data()[1 * 4 + j];

			E_i.data()[(j * 3 + 2) * 6 + 5] = 2*ele->Phi_derivate.data()[2 * 4 + j];
		}
		E_i /= 2.0;

		for (int j = 0; j < 12; j++)
		{
			int nodeid = ele->node_indices[j / 3];
			int col = nodeid * 3 + j%3;
			for (int k = 0; k < 6; k++)
			{
				int row = i * 6 + k;

				tri_entry_.push_back(EIGEN_TRI(row,col,E_i.data()[j*6+k]));
			}
		}
	}
	E->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}



void ModalRotationMatrix::computeRotationStran_To_g(VectorXd &w, VectorXd &e, VectorXd &g)
{
	int numElement = volumtricMesh->getNumElements();
	g.resize(numElement * 9);
	g.setZero();


	for (int i = 0; i < numElement; i++)
	{
		Vector3d w_i;
		w_i.data()[0] = w.data()[i * 3 + 0];
		w_i.data()[1] = w.data()[i * 3 + 1];
		w_i.data()[2] = w.data()[i * 3 + 2];

		double wwNorm = w_i.norm();

		Vector3d w_hat = w_i.normalized();
		if (std::abs(wwNorm) < 1e-15)
		{
			w_hat.setZero();
		}


		double cs = (1 - std::cos(wwNorm));
		double sn = std::sin(wwNorm);



		Matrix3d w_hat_crossMat;
		skewMatrix(w_hat, w_hat_crossMat);

		Matrix3d R_i;
		R_i.setIdentity();
		R_i += w_hat_crossMat*sn + w_hat_crossMat*w_hat_crossMat*cs;

		Matrix3d S_i;
		S_i.setZero();
		S_i.data()[0 * 3 + 0] = e.data()[i * 6 + 0];
		S_i.data()[1 * 3 + 0] = e.data()[i * 6 + 1];
		S_i.data()[2 * 3 + 0] = e.data()[i * 6 + 2];

		S_i.data()[0 * 3 + 1] = e.data()[i * 6 + 1];
		S_i.data()[1 * 3 + 1] = e.data()[i * 6 + 3];
		S_i.data()[2 * 3 + 1] = e.data()[i * 6 + 4];

		S_i.data()[0 * 3 + 2] = e.data()[i * 6 + 2];
		S_i.data()[1 * 3 + 2] = e.data()[i * 6 + 4];
		S_i.data()[2 * 3 + 2] = e.data()[i * 6 + 5];

		S_i += Matrix3d::Identity();
		Matrix3d F = R_i*S_i - Matrix3d::Identity();

		for (int j = 0; j < 9; j++)
		{
			g.data()[i * 9 + j] = F.data()[j];
		}
	}
}

void ModalRotationMatrix::computeRotationStrain_To_g_node(VectorXd &w, VectorXd &e, VectorXd &g)
{
	int numVertex = volumtricMesh->getNumVertices();
	g.resize(numVertex * 9);
	g.setZero();

	for (int i = 0; i < numVertex; i++)
	{
		Vector3d w_i;
		w_i.data()[0] = w.data()[i * 3 + 0];
		w_i.data()[1] = w.data()[i * 3 + 1];
		w_i.data()[2] = w.data()[i * 3 + 2];

		double wwNorm = w_i.norm();

		Vector3d w_hat = w_i.normalized();
		if (std::abs(wwNorm) < 1e-15)
		{
			w_hat.setZero();
		}


		double cs = (1 - std::cos(wwNorm));
		double sn = std::sin(wwNorm);



		Matrix3d w_hat_crossMat;
		skewMatrix(w_hat, w_hat_crossMat);

		Matrix3d R_i;
		R_i.setIdentity();
		R_i += w_hat_crossMat*sn + w_hat_crossMat*w_hat_crossMat*cs;

		AngleAxis<double> aa(wwNorm, w_hat);
		R_i = aa.toRotationMatrix();


		Matrix3d S_i;
		S_i.setZero();
		S_i.data()[0 * 3 + 0] = e.data()[i * 6 + 0];
		S_i.data()[1 * 3 + 0] = e.data()[i * 6 + 1];
		S_i.data()[2 * 3 + 0] = e.data()[i * 6 + 2];

		S_i.data()[0 * 3 + 1] = e.data()[i * 6 + 1];
		S_i.data()[1 * 3 + 1] = e.data()[i * 6 + 3];
		S_i.data()[2 * 3 + 1] = e.data()[i * 6 + 4];

		S_i.data()[0 * 3 + 2] = e.data()[i * 6 + 2];
		S_i.data()[1 * 3 + 2] = e.data()[i * 6 + 4];
		S_i.data()[2 * 3 + 2] = e.data()[i * 6 + 5];

		S_i += Matrix3d::Identity();
		Matrix3d F = R_i*S_i;
		//F = S_i;

		for (int j = 0; j < 9; j++)
		{
			g.data()[i * 9 + j] = F.data()[j];
		}
	}
}

void ModalRotationMatrix::computeModalStrainMatrix(VectorXd &E, double mean , double max, double min)
{
	int numElement = volumtricMesh->getNumElements();
	int numVertex = volumtricMesh->getNumVertices();
	E.resize(numVertex);
	E.setZero();

	for (int i = 0; i < numVertex; i++)
	{
		nodeStrain[i].setZero();
	}

	Matrix3d Ds;
	VectorXd dis(numVertex * 3);
	volumtricMesh->getDisplacement(dis.data());

	for (int i = 0; i < numElement; i++)
	{
		TetElement* ele = volumtricMesh->getTetElement(i);
		//ele->computeElementDeformationshapeMatrix(Ds, dis);
		//F = Ds*ele->Dm_inverse - I;
		Ds.setZero();
		int ne[4];
		for (int i = 0; i < 4; i++)
		{
			ne[i] = ele->node_indices[i];
		}

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				Ds.data()[j * 3 + i] = dis[ne[j] * 3 + i] - dis[ne[3] * 3 + i];
			}
		}

		Ds += ele->Dm;

		Matrix3d gU = Ds*ele->Dm_inverse - Matrix3d::Identity();
		Matrix3d strain = 0.5*(gU + gU.transpose());

		for (int j = 0; j < ele->node_indices.size(); j++)
		{
			int nodeid = ele->node_indices[j];
			LoboNodeBase* node = volumtricMesh->getNodeRef(nodeid);
			int eleCounter = node->element_list.size();
			nodeStrain[nodeid] += strain / eleCounter;
		}
	}

	for (int i = 0; i < numVertex; i++)
	{
		E[i] = (nodeStrain[i].norm()-mean)/(max-min);
	}

}

void ModalRotationMatrix::computeWarpingRotationMatrixR(SparseMatrix<double>* R_, VectorXd w)
{
	int numVertex = volumtricMesh->getNumVertices();
	int R = volumtricMesh->getNumVertices() * 3;
	R_->resize(R, R);
	std::vector<EIGEN_TRI> tri_entry_;

	for (int i = 0; i < numVertex; i++)
	{
		Vector3d ww;
		ww.data()[0] = w.data()[i * 3 + 0];
		ww.data()[1] = w.data()[i * 3 + 1];
		ww.data()[2] = w.data()[i * 3 + 2];

		double wwNorm = ww.norm();

		Vector3d w_hat = ww.normalized();
		if (std::abs(wwNorm) < 1e-5)
		{
			wwNorm += 1;
			w_hat = ww / wwNorm;
		}

		double cs = (1 - std::cos(wwNorm)) / wwNorm;
		double sn = 1 - std::sin(wwNorm) / wwNorm;

		Matrix3d w_hat_crossMat;
		skewMatrix(w_hat, w_hat_crossMat);

		Matrix3d R_i;
		R_i.setIdentity();
		R_i += w_hat_crossMat*cs + w_hat_crossMat*w_hat_crossMat*sn;

		/*AngleAxis<double> aa(wwNorm, w_hat);
		R_i = aa.toRotationMatrix();*/

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				int row = i * 3 + r;
				int col = i * 3 + c;

				double value = R_i.data()[c * 3 + r];
				tri_entry_.push_back(EIGEN_TRI(row,col,value));
			}
		}

	}

	R_->setFromTriplets(tri_entry_.begin(), tri_entry_.end());

}

void ModalRotationMatrix::computeWarpingRotationmatrixR_subnodes(SparseMatrix<double>* R_, VectorXd w, std::vector<int> &nodelist)
{
	int numVertex = volumtricMesh->getNumVertices();
	int R = volumtricMesh->getNumVertices() * 3;
	R_->resize(R, R);
	std::vector<EIGEN_TRI> tri_entry_;

	for (int j = 0; j< nodelist.size(); j++)
	{
		int i = nodelist[j];
		Vector3d ww;
		ww.data()[0] = w.data()[i * 3 + 0];
		ww.data()[1] = w.data()[i * 3 + 1];
		ww.data()[2] = w.data()[i * 3 + 2];

		double wwNorm = ww.norm();

		Vector3d w_hat = ww.normalized();
		if (std::abs(wwNorm) < 1e-5)
		{
			wwNorm += 1;
			w_hat = ww / wwNorm;
		}

		double cs = (1 - std::cos(wwNorm)) / wwNorm;
		double sn = 1 - std::sin(wwNorm) / wwNorm;

		Matrix3d w_hat_crossMat;
		skewMatrix(w_hat, w_hat_crossMat);

		Matrix3d R_i;
		R_i.setIdentity();
		R_i += w_hat_crossMat*cs + w_hat_crossMat*w_hat_crossMat*sn;

		/*AngleAxis<double> aa(wwNorm, w_hat);
		R_i = aa.toRotationMatrix();*/

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				int row = i * 3 + r;
				int col = i * 3 + c;

				double value = R_i.data()[c * 3 + r];
				tri_entry_.push_back(EIGEN_TRI(row, col, value));
			}
		}

	}

	R_->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}

void ModalRotationMatrix::computeWarpingRotationMatrixRi(VectorXd &w, int vertexid, Matrix3d & rotationR)
{
	Vector3d ww;
	ww.data()[0] = w.data()[vertexid * 3 + 0];
	ww.data()[1] = w.data()[vertexid * 3 + 1];
	ww.data()[2] = w.data()[vertexid * 3 + 2];

	double wwNorm = ww.norm();

	Vector3d w_hat = ww.normalized();
	if (std::abs(wwNorm) < 1e-5)
	{
		wwNorm += 1;
		w_hat = ww / wwNorm;
	}

	double cs = (1 - std::cos(wwNorm)) / wwNorm;
	double sn = 1 - std::sin(wwNorm) / wwNorm;

	Matrix3d w_hat_crossMat;
	skewMatrix(w_hat, w_hat_crossMat);

	Matrix3d R_i;
	R_i.setIdentity();
	R_i += w_hat_crossMat*cs + w_hat_crossMat*w_hat_crossMat*sn;

	rotationR = R_i;
}

void ModalRotationMatrix::computeLocalOrientationMatrixR(SparseMatrix<double>* R_, VectorXd w, bool transpose)
{
	int numVertex = volumtricMesh->getNumVertices();
	int R = volumtricMesh->getNumVertices() * 3;
	R_->resize(R, R);
	std::vector<EIGEN_TRI> tri_entry_;

	for (int i = 0; i < numVertex; i++)
	{
		Vector3d ww;
		ww.data()[0] = w.data()[i * 3 + 0];
		ww.data()[1] = w.data()[i * 3 + 1];
		ww.data()[2] = w.data()[i * 3 + 2];

		double wwNorm = ww.norm();

		Vector3d w_hat = ww.normalized();
		if (std::abs(wwNorm) < 1e-15)
		{
			w_hat.setZero();
		}

		double cs = (1 - std::cos(wwNorm));
		double sn = std::sin(wwNorm);

		Matrix3d w_hat_crossMat;
		skewMatrix(w_hat, w_hat_crossMat);

		Matrix3d R_i;
		R_i.setIdentity();
		R_i += w_hat_crossMat*sn + w_hat_crossMat*w_hat_crossMat*cs;

	

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				int row = i * 3 + r;
				int col = i * 3 + c;

				double  value = 0;

				if (transpose)
					value = R_i.data()[r * 3 + c];
				else
					value = R_i.data()[c * 3 + r];
				tri_entry_.push_back(EIGEN_TRI(row, col, value));
			}
		}

	}

	R_->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}

void ModalRotationMatrix::computeLocalOrientationByPolarDecomposition(SparseMatrix<double>* R_, VectorXd q, bool transpose /*= true*/)
{
	int numVertex = volumtricMesh->getNumVertices();
	int R = volumtricMesh->getNumVertices() * 3;
	R_->resize(R, R);
	std::vector<EIGEN_TRI> tri_entry_;

	volumtricMesh->setDisplacement(q.data());
	for (int i = 0; i < numVertex; i++)
	{
		Matrix3d R_i;
		volumtricMesh->computeNodeRotationRing(i, R_i);

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				int row = i * 3 + r;
				int col = i * 3 + c;

				double  value = 0;

				if (transpose)
					value = R_i.data()[r * 3 + c];
				else
					value = R_i.data()[c * 3 + r];
				tri_entry_.push_back(EIGEN_TRI(row, col, value));
			}
		}
	}
	R_->setFromTriplets(tri_entry_.begin(), tri_entry_.end());
}
