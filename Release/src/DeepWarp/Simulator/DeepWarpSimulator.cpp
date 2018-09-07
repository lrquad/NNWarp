#include "DeepWarpSimulator.h"
#include "Simulator/PoseDataSet/PoseDataSet.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "Simulator/ForceModel/LinearSTVKForceModel.h"
#include "Simulator/STVK/InvertibleSTVKModel.h"
#include "Render/MouseProjection.h"
#include "Functions/findElementInVector.h"
#include "Simulator/DeepWarp/ModalRotationMatrix.h"
#include "Integrator/ImplicitNewMatkSparseIntegrator.h"
#include "Integrator/ImplicitBackwardEulerSparseIntegrator.h"
#include <fstream>
#include <iostream>
#include "Reduced/SubspaceModesCreator.h"
#include "Simulator/ForceModel/STVKForceModel.h"
#include "Simulator/STVK/STVKModel.h"
#include "EigenMatrixIO/EigenMatrixIO.h"
#include <random>
#include "LoboNeuralNetwork/LoboNeuralNetwork.h"
#include "Simulator/DeepWarp/WarpingMapTrainingModel.h"
#include "Functions/GeoMatrix.h"
#include "Simulator/DeepWarp/UniformSamplingTrainingModel.h"
#include "Simulator/DeepWarp/ModalWarpingTrainingModel.h"
#include "Simulator/DeepWarp/SmartTrainingModel.h"
#include "Simulator/DeepWarp/LinearMainSmartTrainingModel.h"
#include "Integrator/ImplicitNewMarkDNNIntegrator.h"
#include "TeMeshVoxlizeModel.h"
#include "Simulator/DeepWarp/TrainingDataGenerator.h"
#include "Simulator/DeepWarp/RotationTrainingDataGenerator.h"
#include "Simulator/DeepWarp/RotationTrainingDataGeneratorV2.h"
#include "Simulator/DeepWarp/RotationTrainingDataGeneratorV3.h"
#include "Simulator/DeepWarp/RotationTrainingDataGeneratorV4.h"
#include "Simulator/DeepWarp/RotationStrainDataGenerator.h"
#include "Simulator/DeepWarp/FullfeatureDataGenerator.h"

#include "Simulator/IsotropicHyperelasticFEM/isotropicHyperelasticCore.h"
#include "Simulator/ForceModel/IsotropicHyperlasticForceModel.h"
#include "Integrator/ImplicitModalWarpingIntegrator.h"
#include "Simulator/ForceField/RotateForceField.h"
#include "Functions/GeoMatrix.h"
#include "Functions/SkewMatrix.h"
#include "SparseMatrix/SparseMatrixRemoveRows.h"
#include "Functions/cnpy.h"
#include "Simulator/corotational/CorotationalModel.h"



DeepWarpSimulator::DeepWarpSimulator(std::ifstream &readconfig_, LoboTriMesh* obj_mesh, bool ifreadconfig) :LoboSimulatorBase(readconfig_, obj_mesh)
{
	nullAllPointer();
	if (ifreadconfig)
	{
		readConfig(readconfig_);
		if (volumetric_mesh != NULL)
		{
			selectedNodes.reserve(volumetric_mesh->getNumVertices());
		}
	}
}

DeepWarpSimulator::~DeepWarpSimulator()
{
	deleteAllPointer();
}

void DeepWarpSimulator::nullAllPointer()
{
	nodeNeighborTypeExport = 0;
	modesk = 3;
	modesr = 10;
	showModeIndex = 0;
	massMatrix = NULL;
	invertibleStvkModel = NULL;
	stvkforce_model = NULL;
	trainingModal = NULL;
	cubeVolumtricMesh = NULL;
	dataGenerator = NULL;
	cubeVolumtricMesh_distace = NULL;
	nonlinearMapping = true;
	modalwarpingMapping = false;
	seperateGenerateData = false;
	finishedInit = false;
	modalRotationSparseMatrix = new SparseMatrix<double>();
	rotationSparseMatrixR = new SparseMatrix<double>();
	localOrientationMatrixR = new SparseMatrix<double>();

	G = new SparseMatrix<double>();
	V = new SparseMatrix<double>();
	e_ele = new SparseMatrix<double>();
	w_ele = new SparseMatrix<double>();

	DNNpath = "data/DNN/test2.txt";
	selectedNodeInput.resize(8);
	selectedNodeInput.setZero();
	setGPUversion(false);

}

void DeepWarpSimulator::deleteAllPointer()
{
	delete G;
	delete V;
	delete e_ele;
	delete w_ele;

	delete massMatrix;
	delete invertibleStvkModel;
	delete stvkforce_model;
	delete modalRotationSparseMatrix;
	delete rotationSparseMatrixR;
	delete localOrientationMatrixR;
	delete trainingModal;
	delete cubeVolumtricMesh;
	delete cubeVolumtricMesh_distace;

	delete hyperforcemodel;
	delete isotropicMaterial;
	delete isotropicHyperelasticModel;
	delete dataGenerator;

	delete loboNeuralNetwork;
	delete loboNeuralNetworkList[0];
	delete loboNeuralNetworkList[1];
	delete loboNeuralNetworkList[2];
	delete loboNeuralNetworkList[3];



}

void DeepWarpSimulator::initSimulator(int verbose /*= 0*/)
{

	domainscale = volumetric_mesh->computeUniformScaleExceptConstraint(numConstrainedDOFs, constrainedDOFs);

	int numVertex = volumetric_mesh->getNumVertices();
	int R = numVertex * 3;
	runtimePoseDataSet = new PoseDataSet(R);

	loboNeuralNetwork = new LoboNeuralNetwork();
	loboNeuralNetwork->loadDNNV2(DNNpath.c_str());
	//loboNeuralNetwork->convertToGLSL("GLSLdata.txt");

	//loboNeuralNetworkList.resize(4);
	//loboNeuralNetworkList[0] = new LoboNeuralNetwork();
	//loboNeuralNetworkList[0]->loadDNNV2("./data/DNN/type0.txt","tanh");
	//loboNeuralNetworkList[1] = new LoboNeuralNetwork();
	//loboNeuralNetworkList[1]->loadDNNV2("./data/DNN/type1.txt", "tanh");
	//loboNeuralNetworkList[2] = new LoboNeuralNetwork();
	//loboNeuralNetworkList[2]->loadDNNV2("./data/DNN/type2.txt", "tanh");
	//loboNeuralNetworkList[3] = new LoboNeuralNetwork();
	//loboNeuralNetworkList[3]->loadDNNV2("./data/DNN/type3.txt", "tanh");

	externalForce.resize(R);
	externalForce.setZero();

	mouseForce.resize(R);
	mouseForce.setZero();

	collisionExternalForce.resize(R);
	collisionExternalForce.setZero();

	//tetmeshVoxModel = new TeMeshVoxlizeModel(cubeVolumtricMesh,downCastTetVolMesh(volumetric_mesh));
	//tetmeshVoxModel->createCubeTetMapping();

	((TetVolumetricMesh*)volumetric_mesh)->initElementShapeFunctionDerivate();
	volumetric_mesh_graph = new LoboVolumetricMeshGraph(volumetric_mesh);
	volumetric_mesh_graph->init();

	initVolumnMeshTriMeshAdapter();

	massMatrix = new SparseMatrix<double>(R, R);
	volumetric_mesh->computeMassMatrix(massMatrix);

	invertibleStvkModel = new InvertibleSTVKModel(this->volumetric_mesh, massMatrix, 0.5, 1, 500);
	invertibleStvkModel->precomputedFdU();
	invertibleStvkModel->setGravity(getUseGravity());
	invertibleStvkModel->setGravityDirection(gravityDirection);

	stvkforce_model = new LinearSTVkForceModel(invertibleStvkModel);

	stvkforceNonlinear_model = new STVKForceModel(invertibleStvkModel);

	modalrotationMatrix = new ModalRotationMatrix(downCastTetVolMesh(volumetric_mesh));

	modalrotationMatrix->computeModalRotationSparseMatrix_W(modalRotationSparseMatrix);


	//method 2
	((TetVolumetricMesh*)volumetric_mesh)->computeDeformationOperator(G);
	((TetVolumetricMesh*)volumetric_mesh)->computeElementVolumeMatrix(V);
	modalrotationMatrix->computeModalRotationSparseMatrix_w_perele(w_ele);
	modalrotationMatrix->computeModalRotationSparseMatrix_E_perele(e_ele);

	modalrotationMatrix->computeModalRotationSparseMatrix_E(modalrotationMatrix->e_operator);
	modalrotationMatrix->computeModalRotationSparseMatrix_W(modalrotationMatrix->w_operator);


	//rotationStrainMatrix = G->transpose()*(*V)*(*G);

	//createMapByConstrains(mapOldNew, R, numConstrainedDOFs, constrainedDOFs);
	//createSparseMapbyTopology(&rotationStrainMatrix, &subRotationStrainMatrix, mapSparseMatrixEntryOldNew, mapOldNew, R, numConstrainedDOFs, constrainedDOFs);

	//LDLT.compute(subRotationStrainMatrix);


	//init neohooken
	isotropicMaterial = new LoboneoHookeanIsotropicMaterial(downCastTetVolMesh(volumetric_mesh), 0, 500);
	//isotropicMaterial = new LoboStVKIsotropicMaterial(downCastTetVolMesh(volumetric_mesh), 1, 500);

	isotropicHyperelasticModel = new CorotationalModel(downCastTetVolMesh(volumetric_mesh), isotropicMaterial, massMatrix, 0.5, false, 500.0);
	isotropicHyperelasticModel->setIsInvertible(false);
	isotropicHyperelasticModel->setAddGravity(false);
	hyperforcemodel = new IsotropicHyperlasticForceModel(isotropicHyperelasticModel);

	forcefield = new RotateForceField(volumetric_mesh, Vector3d(-1, 0.0, 0));
	forcefield->setForceMagnitude(0.001);

	/*std::ofstream test("test.txt");
	test << *massMatrix << std::endl;
	test.close();
	*/
	//initSubspaceModes();
	initTrainingDataModel();

	initIntegrator();
	getNodePotentialValue();
	//std::ifstream test("extforce.txt");
	//int dimension;
	//test >> dimension;
	//VectorXd g(dimension);
	//for (int i = 0; i < dimension; i++)
	//{
	//	test >> g.data()[i];
	//}
	//invertibleStvkModel->setGravityVector(g);
	
	
	//initCollisionHandle();
	//simulationstate = LoboSimulatorBase::ready;
	finishedInit = true;

	generateEnergyCurveData();

	//generateLinearRotationTrainingData();

}

void DeepWarpSimulator::updateSimulator(int verbose /*= 0*/)
{
	mergeExternalForce();

	integrator->doTimeStep();

	int numVertex = volumetric_mesh->getNumVertices();
	int R = numVertex * 3;

	VectorXd fullq(R);
	memcpy(fullq.data(), integrator->getq(), R*sizeof(double));

	//showModeIndex = (simulation_steps / 100) % reduced_modes.cols();

	//fullq = reduced_modes.col(showModeIndex)*std::sin(simulation_steps*0.1) * 0.5;

	//warping here

	VectorXd w = (*modalRotationSparseMatrix)*fullq;
	
	if (0)
	for (int i = 0; i < numVertex; i++)
	{
		VectorXd inputV(3);
		inputV.data()[0] = w.data()[i * 3 + 0];
		inputV.data()[1] = w.data()[i * 3 + 1];
		inputV.data()[2] = w.data()[i * 3 + 2];
		

		VectorXd rotaionmatrix = loboNeuralNetwork->predict(inputV);
		Matrix3d rotationMatrix_;
		for (int j = 0; j < 9; j++)
		{
			rotationMatrix_.data()[j] = rotaionmatrix.data()[j];
		}
		
		Vector3d oridis;
		for (int j = 0; j < 3; j++)
		{
			oridis.data()[j] = fullq.data()[i * 3 + j];
		}
		oridis = rotationMatrix_*oridis;
		for (int j = 0; j < 3; j++)
		{
			fullq.data()[i * 3 + j] = oridis.data()[j];
		}
	}

	if (0)
	for (int i = 0; i < numVertex; i++)
	{
		VectorXd inputV(12);
		Matrix3d R_;
		modalrotationMatrix->computeWarpingRotationMatrixRi(w, i, R_);
		for (int j = 0; j < 9; j++)
		{
			inputV.data()[j + 3] = R_.data()[j];
		}
		/*inputV.data()[3] = w.data()[i * 3 + 0];
		inputV.data()[4] = w.data()[i * 3 + 1];
		inputV.data()[5] = w.data()[i * 3 + 2];*/

		inputV.data()[0] = fullq.data()[i * 3 + 0];
		inputV.data()[1] = fullq.data()[i * 3 + 1];
		inputV.data()[2] = fullq.data()[i * 3 + 2];

		VectorXd oridis = loboNeuralNetwork->predictV2(inputV);

		for (int j = 0; j < 3; j++)
		{
			fullq.data()[i * 3 + j] += oridis.data()[j];
		}
	}

	if (0)
	for (int i = 0; i < numConstrainedDOFs; i++)
	{
		fullq.data()[constrainedDOFs[i]] = 0;
	}

	if (getNonlinearMapping())
	{
		if (getGPUversion())
		{
			VectorXd dnnInput;

			int dimension = dataGenerator->getInputDimension()+9;

			((RotationStrainDataGenerator*)dataGenerator)->createInputDataWithRotation(fullq, dnnInput, invertibleStvkModel->getGravity());
			this->volumetric_obj_adapter->sendVertexDataToTriMesh(dnnInput, dimension);

		}else
		if (getModalwarpingMapping())
		{

			//modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);
			modalrotationMatrix->computeWarpingRotationmatrixR_subnodes(rotationSparseMatrixR, w, volumetric_mesh->surface_node_);
			fullq = (*rotationSparseMatrixR)*fullq;
			
			/*VectorXd w = *w_ele*fullq;
			VectorXd e = *e_ele*fullq;
			VectorXd g;
			modalrotationMatrix->computeRotationStran_To_g(w, e, g);


			VectorXd rhs = (G->transpose()**V)*g;
			VectorXd subrhs(R - numConstrainedDOFs);
			VectorRemoveRows(mapOldNew, rhs, subrhs, numConstrainedDOFs, constrainedDOFs);

			VectorXd subfullq = LDLT.solve(subrhs);

			VectorInsertRows(mapOldNew, subfullq, fullq, numConstrainedDOFs, constrainedDOFs);*/

		}
		else
		{
			VectorXd dnnInput;

			int dimension = dataGenerator->getInputDimension();

			/*VectorXd extforce = ((ImplicitModalWarpingIntegrator*)integrator)->getInteranlForce(fullq);
			modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
			extforce = *localOrientationMatrixR*extforce;*/

			if (((RotationStrainDataGenerator*)dataGenerator)->getInputtype() == 0)
			{
				((RotationStrainDataGenerator*)dataGenerator)->createInputData(fullq, dnnInput, invertibleStvkModel->getGravity());
			}
			else
			{
				VectorXd forceaxis(numVertex * 3);
				forceaxis.setZero();
				forceaxis.data()[0] = forcefield->centeraxis.data()[0];
				forceaxis.data()[1] = forcefield->centeraxis.data()[1];
				forceaxis.data()[2] = forcefield->centeraxis.data()[2];
				((RotationStrainDataGenerator*)dataGenerator)->createInputData(fullq, dnnInput, forceaxis);
			}

			for (int l = 0; l < volumetric_mesh->surface_node_.size(); l++)
			{
				int i = volumetric_mesh->surface_node_[l];

				VectorXd inputV(dimension);
				for (int j = 0; j < dimension; j++)
				{
					inputV.data()[j] = dnnInput.data()[i * dimension + j];
				}

				//inputV.data()[3] /= domainscale;

				/*if (std::abs(inputV.data()[3]) < 0.005)
				{
					continue;
				}*/

				//int cubeid = tetmeshVoxModel->getNodeCube(i);
				//CubeElement* element = cubeVolumtricMesh->getCubeElement(cubeid);
				//CubeNodeMapType* cubenodemap = tetmeshVoxModel->getCubeNodeMapRefer(i);

				int nodetype = 0;
				
				/*if (getSeperateGenerateData())
				nodetype = cubenodemap->getMainType();*/

				VectorXd oridis = loboNeuralNetwork->predictV2(inputV);

				//VectorXd oridis = loboNeuralNetwork->predictV2(inputV);
				//oridis *= domainscale;

				Vector3d dis_;
				for (int j = 0; j < 3; j++)
				{
					dis_.data()[j] = oridis.data()[j];
				}

				((RotationTrainingDataGeneratorV3*)dataGenerator)->convertOutput(dis_, i);

				for (int j = 0; j < 3; j++)
				{
					fullq.data()[i * 3 + j] = fullq.data()[i * 3 + j] + dis_.data()[j];
				}
			}

			for (int i = 0; i < numConstrainedDOFs; i++)
			{
				fullq.data()[constrainedDOFs[i]] = 0;
			}
		}
	}

	//volumetric_mesh->setDisplacement(fullq.data());
	updateTriAndVolumeMesh(fullq.data());

	//generate data
	if (0)
	if (simulation_steps == 0)
	{
		std::vector<double> origin_data;
		std::vector<double> target_data;

		std::vector<double> test_origin_data;
		std::vector<double> test_target_data;

		std::vector<double> axis_list;

		generatePolarDecompositionData(
			M_PI * 2, 0, 20,
			M_PI / 2, -M_PI / 2, 20,
			M_PI*1.5, -M_PI*1.5, 20,
			1, -1, 50,
			origin_data,
			target_data,
			axis_list
			);

		//trainingModal->exportExampleData("axis_list.txt", axis_list.data(), 5, axis_list.size());
		
		
		/*generatePolarDecompositionData(
			M_PI * 2, 0, 2,
			M_PI / 2, -M_PI / 2, 2,
			M_PI, -M_PI, 100,
			1, -1, 50,
			test_origin_data,
			test_target_data,
			axis_list
			);*/
		

		generatePolarDecompositionDataRandomly(500, test_origin_data, test_target_data, axis_list);

		/*trainingModal->exportExampleData("axis_list.txt", axis_list.data(), 4, axis_list.size());*/

		trainingModal->exportExampleData("origin_train.txt", origin_data.data(), 9, origin_data.size());
		trainingModal->exportExampleData("target_train.txt", target_data.data(), 9, target_data.size());

		trainingModal->exportExampleData("origin_test.txt", test_origin_data.data(), 9, test_origin_data.size());
		trainingModal->exportExampleData("target_test.txt", test_target_data.data(), 9, test_target_data.size());



		system("pause");
	}


	if (0)
	if (simulation_steps == 0)
	{
		int numData = 600;

		std::vector<double> origin_data;
		std::vector<double> target_data;
		VectorXd randomforce(numVertex * 3);

		//LDLT.compute(*stiffness);


		for (int i = 0; i < numData; i++)
		{
			srand(time(NULL) + i);
			fullq.setRandom();
			double scale_ = (double)(std::rand()%1000)/100+1.0;
			fullq *= scale_;

			randomforce.setRandom();
			randomforce *= scale_;



			VectorXd w = (*modalRotationSparseMatrix)*fullq;

			for (int j = 0; j < numVertex; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					origin_data.push_back(w.data()[j * 3 + k]);
				}
				for (int k = 0; k < 3; k++)
				{
					origin_data.push_back(fullq.data()[j * 3 + k]);
				}
			}


			modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);

			fullq = (*rotationSparseMatrixR)*fullq;

			/*Matrix3d rotationMatrix_;
			for (int j = 0; j < numVertex; j++)
			{
			modalrotationMatrix->computeWarpingRotationMatrixRi(w, j, rotationMatrix_);
			for (int k = 0; k < 9; k++)
			{
			target_data.push_back(rotationMatrix_.data()[k]);
			}
			}*/
			for (int j = 0; j < fullq.size(); j++)
			{
				target_data.push_back(fullq.data()[j]);
			}

		}

		exportFeatureData("origin_train.txt", origin_data, 6);
		exportTargetData("target_train.txt", target_data, 3);

		system("pause");
	}

	/*int nodeid = 1036;
	Vector3d nodep = volumetric_mesh->getNodeDisplacement(nodeid);
	nodetrajectory.push_back(nodep.norm());

	if (simulation_steps == 1000)
	{
	std::ofstream outputStream("nodetrajectory.txt");
	int numdata = nodetrajectory.size();
	for (int i = 0; i < numdata; i++)
	{
	outputStream << nodetrajectory.data()[i] << std::endl;
	}
	outputStream.close();
	setSimulatorStop();
	}*/

	this->simulation_steps++;

}

void DeepWarpSimulator::resetSimulator(int verbose /*= 0*/)
{
	runtimePoseDataSet->clearData();
	integrator->resetToRestNoClearSequence();
	
	int numVertex = volumetric_mesh->getNumVertices();
	int R = numVertex * 3;

	VectorXd fullq(R);
	memcpy(fullq.data(), integrator->getq(), R*sizeof(double));
	updateTriAndVolumeMesh(fullq.data());
	
	nodetrajectory.clear();

	simulation_steps = 0;
}

LoboSimulatorBase::simulatorType DeepWarpSimulator::getType()
{
	return LoboSimulatorBase::MODALWARPING;
}

void DeepWarpSimulator::saveSimulator(const char* filename, fileFormatType formattype) const
{

}

void DeepWarpSimulator::readSimulator(const char* filename, fileFormatType formattype)
{

}

void DeepWarpSimulator::readVolumetricMeshAscii(const char* filenamebase)
{
	TetVolumetricMesh* tet_volumetric_mesh = new TetVolumetricMesh();

	Vector3d translate = Vector3d::Zero();

	tet_volumetric_mesh->setUniformMeshAfterLoad(getUniformVolumtricMesh());
	tet_volumetric_mesh->setUnifromUseScaleEx(false);

	tet_volumetric_mesh->readElementMesh(filenamebase, simulator_translate.data(), 1);

	volumetric_mesh = tet_volumetric_mesh;

	LoboVolumetricMesh::Material* materia = volumetric_mesh->getMaterialById(0);
	LoboVolumetricMesh::ENuMaterial* enmateria = (LoboVolumetricMesh::ENuMaterial*)materia;
	enmateria->setE(this->youngmodulus);
	enmateria->setNu(this->poisson);
	enmateria->setDensity(this->density);

	if (getUniformVolumtricMesh())
	{
		Vector3d tritranslate = -simulator_translate - volumetric_mesh->getUniformT();
		this->tri_mesh->translateMesh(tritranslate);
		this->tri_mesh->scaleAtCurrentUpdate(1.0 / volumetric_mesh->getUniformScale());
	}

}

void DeepWarpSimulator::readCubeMeshAscii(const char* filenamebase)
{
	cubeVolumtricMesh = new CubeVolumtricMesh();
	Vector3d translate = Vector3d::Zero();

	double scale_ = 1.0/volumetric_mesh->getUniformScale();
	translate = -volumetric_mesh->getUniformT();

	if (!getUniformVolumtricMesh())
	{
		scale_ = 1.0;
		translate.setZero();
	}

	cubeVolumtricMesh->setUniformMeshAfterLoad(false);

	cubeVolumtricMesh->setUnifromUseScaleEx(true);
	cubeVolumtricMesh->readElementMesh(filenamebase, translate.data(), scale_);

}

void DeepWarpSimulator::creadCubeMeshDistanceAscii(const char* filenamebase)
{

	cubeVolumtricMesh_distace = new CubeVolumtricMesh();
	Vector3d translate = Vector3d::Zero();

	double scale_ = 1.0 / volumetric_mesh->getUniformScale();
	translate = -volumetric_mesh->getUniformT();

	if (!getUniformVolumtricMesh())
	{
		scale_ = 1.0;
		translate.setZero();
	}

	cubeVolumtricMesh_distace->setUniformMeshAfterLoad(false);

	cubeVolumtricMesh_distace->setUnifromUseScaleEx(true);
	cubeVolumtricMesh_distace->readElementMesh(filenamebase, translate.data(), scale_);
}

void DeepWarpSimulator::exportConvexCubeMesh(const char* filenamebase)
{
	cubeVolumtricMesh->exportNewCubeVolumtricMeshByNode(filenamebase);
}

void DeepWarpSimulator::initSimulatorGlobalConstraints()
{
	if (constrainedDOFs != NULL)
	{
		free(constrainedDOFs);
	}
	int numVertex = volumetric_mesh->getNumVertices();

	std::vector<bool> nodemark(numVertex);
	std::fill(nodemark.begin(), nodemark.end(), false);

	for (int i = 0; i < selectedNodes.size(); i++)
	{
		int nodeid = selectedNodes[i];
		nodemark[nodeid] = true;
	}

	constrainedNodes.clear();
	for (int i = 0; i < numVertex; i++)
	{
		if (nodemark[i])
		{
			constrainedNodes.push_back(i);
		}
	}

	numConstrainedDOFs = constrainedNodes.size() * 3;
	constrainedDOFs = (int*)malloc(sizeof(int) * numConstrainedDOFs);
	for (int i = 0; i < numConstrainedDOFs / 3; i++)
	{
		constrainedDOFs[i * 3 + 0] = constrainedNodes[i] * 3 + 0;
		constrainedDOFs[i * 3 + 1] = constrainedNodes[i] * 3 + 1;
		constrainedDOFs[i * 3 + 2] = constrainedNodes[i] * 3 + 2;
	}

	std::cout << "created constrain: size " << numConstrainedDOFs << std::endl;
}

void DeepWarpSimulator::getNodeOrientation(VectorXd &nodeYaxis)
{
	int numVertex = volumetric_mesh->getNumVertices();

	nodeYaxis.resize(numVertex * 3);
	nodeYaxis.setZero();
	for (int i = 0; i < numVertex; i++)
	{
		nodeYaxis.data()[i * 3 + 1] = 1;
	}

	VectorXd q(numVertex * 3);
	volumetric_mesh->getDisplacement(q.data());
	VectorXd lq = integrator->getVectorq();
	lq = q;
	VectorXd w = *modalRotationSparseMatrix*q;
	//modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	modalrotationMatrix->computeLocalOrientationByPolarDecomposition(localOrientationMatrixR, lq, false);
	nodeYaxis = *localOrientationMatrixR*nodeYaxis;
	volumetric_mesh->setDisplacement(q.data());

}

void DeepWarpSimulator::getNodeOrientation(VectorXd &nodeYaxis, Vector3d axis)
{
	int numVertex = volumetric_mesh->getNumVertices();

	nodeYaxis.resize(numVertex * 3);
	nodeYaxis.setZero();
	for (int i = 0; i < numVertex; i++)
	{
		nodeYaxis.data()[i * 3 + 0] = axis.data()[0];
		nodeYaxis.data()[i * 3 + 1] = axis.data()[1];
		nodeYaxis.data()[i * 3 + 2] = axis.data()[2];
	}

	VectorXd q(numVertex * 3);
	volumetric_mesh->getDisplacement(q.data());
	VectorXd lq = integrator->getVectorq();
	lq = q;
	VectorXd w = *modalRotationSparseMatrix*q;
	modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	//modalrotationMatrix->computeLocalOrientationByPolarDecomposition(localOrientationMatrixR, lq, false);
	nodeYaxis = *localOrientationMatrixR*nodeYaxis;

	VectorXd extforce;
	//trainingModal->getTraingLinearForce(disIndex,extforce);
	trainingModal->getTraingLinearForce(lq, extforce);

	modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	//extforce = *localOrientationMatrixR*extforce;

	nodeYaxis = extforce;
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d nodeforcedirection;
		nodeforcedirection.data()[0] = nodeYaxis.data()[i * 3 + 0];
		nodeforcedirection.data()[1] = nodeYaxis.data()[i * 3 + 1];
		nodeforcedirection.data()[2] = nodeYaxis.data()[i * 3 + 2];
		nodeforcedirection.normalize();

		nodeYaxis.data()[i * 3 + 0] = nodeforcedirection.data()[0];
		nodeYaxis.data()[i * 3 + 1] = nodeforcedirection.data()[1];
		nodeYaxis.data()[i * 3 + 2] = nodeforcedirection.data()[2];
	}

	volumetric_mesh->setDisplacement(q.data());
}

void DeepWarpSimulator::singleClickEvent()
{
	int numVertex = volumetric_mesh->getNumVertices();
	Vector2d mousescreen;
	mousescreen.x() = mouseprojection->getPreMouseX();
	mousescreen.y() = mouseprojection->getPreMouseY();

	double mindistance = DBL_MAX;
	double mindepth = DBL_MAX;

	int nodeid = -1;

	std::vector<int> surfaceNodes = volumetric_mesh->getSurface_node();
	numVertex = surfaceNodes.size();

	for (int i = 0; i < numVertex; i++)
	{
		int node_id = surfaceNodes[i];
		Vector3d nodecur = volumetric_mesh->getNodePosition(node_id);
		Vector2d screenpos;
		double depth;
		mouseprojection->projectWorldToScreenDepth(nodecur, screenpos, depth);

		Vector2d distance2d = mousescreen - screenpos;

		Vector3d distanceV3(distance2d.x(), distance2d.y(), 10 * depth);

		double distance = distanceV3.norm();

		if (distance <= mindistance)
		{
			mindistance = distance;
			nodeid = node_id;
		}
	}

	std::cout << "selected: " << nodeid << std::endl;
	std::cout << volumetric_mesh->getNodePosition(nodeid).transpose() << std::endl;

	if (trainingModal != NULL)
	{

		VectorXd fullq(numVertex * 3);
		fullq.setZero();
		fullq = this->integrator->getVectorq();
		VectorXd inputDNN;


		((RotationStrainDataGenerator*)dataGenerator)->createInputData(fullq, inputDNN, invertibleStvkModel->getGravity());

		int dimension = dataGenerator->getInputDimension();
		for (int i = 0; i < dimension; i++)
		{
			selectedNodeInput.data()[i] = inputDNN.data()[nodeid*dimension + i];
		}
		selectedNodeid = nodeid;

		std::ofstream test("selectedNodeinput.txt");
		test << selectedNodeInput.transpose() << std::endl;
		test.close();
	}

	mouseprojection->setBindedNodeid(nodeid);

	singleSelectedNeighborNodes.clear();
	neighbordistance.clear();
	Vector3d selectedPosition = volumetric_mesh->getNodePosition(nodeid);;
	for (int i = 0; i < volumetric_mesh->surface_node_.size(); i++)
	{
		int id = volumetric_mesh->surface_node_[i];
		Vector3d nodep = volumetric_mesh->getNodePosition(id);
		if ((nodep - selectedPosition).norm() < 0.1)
		{
			singleSelectedNeighborNodes.push_back(id);
			neighbordistance.push_back((nodep - selectedPosition).norm());
		}
	}


	if (!findElement(loaded_nodeid, nodeid))
	{
		this->loaded_nodeid.push_back(nodeid);
		this->force_on_node.push_back(Vector3d::Zero());
	}

	currentforceid = loaded_nodeid.size() - 1;
}

void DeepWarpSimulator::rectSelectEvent()
{
	int numVertex = volumetric_mesh->getNumVertices();
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d nodecur = volumetric_mesh->getNodePosition(i);
		Vector2d screenpos;
		double depth;
		mouseprojection->projectWorldToScreen(nodecur, screenpos, depth);
		if (mouseprojection->isInMouseRect(screenpos.x(), screenpos.y()))
		{
			selectedNodes.push_back(i);
		}
	}
}

void DeepWarpSimulator::mouseMoveEvent()
{
	if (simulationstate == LoboSimulatorBase::beforeinit)
	{
		return;
	}

	int numVertex = volumetric_mesh->getNumVertices();
	Vector2d mousescreen;
	mousescreen.x() = mouseprojection->getMouseX();
	mousescreen.y() = mouseprojection->getMouseY();
	int bindedNodeid = mouseprojection->getBindedNodeid();
	Vector3d nodecur = volumetric_mesh->getNodePosition(bindedNodeid);
	Vector2d nodescreen;
	double depth;
	mouseprojection->projectWorldToScreen(nodecur, nodescreen, depth);



	Vector3d mouseWorld;
	mouseprojection->projectScreenToWorld(mousescreen, depth, mouseWorld);
	Vector3d force = (mouseWorld - nodecur)*mouseForceRatio;
	//force.data()[1] *= 0.1;


	mouseForce.setZero();
	mouseForce.data()[bindedNodeid * 3 + 0] = force.data()[0];
	mouseForce.data()[bindedNodeid * 3 + 1] = force.data()[1];
	mouseForce.data()[bindedNodeid * 3 + 2] = force.data()[2];

	for (int i = 0; i < singleSelectedNeighborNodes.size(); i++)
	{
		int nodeid = singleSelectedNeighborNodes[i];
		double distance = neighbordistance[i];
		double weight = std::exp(-(distance*distance) / 2.0);
		mouseForce.data()[nodeid * 3 + 0] = force.data()[0] * weight;
		mouseForce.data()[nodeid * 3 + 1] = force.data()[1] * weight;
		mouseForce.data()[nodeid * 3 + 2] = force.data()[2] * weight;
	}



	force_on_node[currentforceid] = force;

	//computeExternalForce(mouseForce.data());
}

void DeepWarpSimulator::mouseReleaseEvent()
{
	this->cleanExternalForce();
	mouseprojection->setBindedNodeid(-1);
}

void DeepWarpSimulator::initSubspaceModes()
{

	/*EigenMatrixIO::read_binary("data/SubspaceModes/longbar_20.txt", reduced_modes);
	return;*/

	STVKModel* stvkmodel = new STVKModel(this->volumetric_mesh, massMatrix);
	stvkmodel->setGravity(false);

	STVKForceModel* stvkforce_model_ = new STVKForceModel(stvkmodel);

	SubspaceModesCreator* modesCreater = new SubspaceModesCreator(stvkforce_model_, massMatrix, numConstrainedDOFs, constrainedDOFs, modesk, modesr);
	reduced_modes = *(modesCreater->getFinalModes());
	reduced_modes = *(modesCreater->getLinearModes());
	//EigenMatrixIO::write_binary("data/SubspaceModes/longbar_20.txt", reduced_modes);
	
	delete stvkmodel;
	delete modesCreater;
	delete stvkforce_model_;
}

void DeepWarpSimulator::readSubspaceModes(const char* filename)
{
	EigenMatrixIO::read_binary(filename, reduced_modes);
}

void DeepWarpSimulator::saveSubspaceModes(const char* filename)
{
	EigenMatrixIO::write_binary(filename, reduced_modes);
}

void DeepWarpSimulator::initTrainingDataModel()
{
	int r = volumetric_mesh->getNumVertices() * 3;
	/*trainingModal = new WarpingMapTrainingModel(&reduced_modes, r, timestep, massMatrix, stvkforce_model, hyperforcemodel, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef);*/

	/*trainingModal = new LinearMainSmartTrainingModel(invertibleStvkModel->getGravity(), &reduced_modes, r, timestep, massMatrix, stvkforce_model, hyperforcemodel, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef,false);*/

	
	//trainingModal = new SmartTrainingModel(invertibleStvkModel->getGravity(), &reduced_modes, r, timestep, massMatrix, stvkforce_model, hyperforcemodel, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef);
	
	trainingModal = new ModalWarpingTrainingModel(volumetric_mesh, invertibleStvkModel->getGravity(), modalRotationSparseMatrix, modalrotationMatrix, &reduced_modes, r, 0.0005, massMatrix, stvkforce_model, hyperforcemodel, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef, false);
	trainingModal->setForcefieldType(0); //0 gravity, 1 rotated force

	dataGenerator = new RotationStrainDataGenerator(tetmeshVoxModel, cubeVolumtricMesh, volumetric_mesh, trainingModal, modalrotationMatrix, modalRotationSparseMatrix, numConstrainedDOFs, constrainedDOFs);
	((RotationTrainingDataGeneratorV3*)dataGenerator)->setInputtype(0);
	//((RotationTrainingDataGeneratorV3*)dataGenerator)->setForcefieldType(1);

	LoboVolumetricMeshGraph* graph = new LoboVolumetricMeshGraph(volumetric_mesh);
	graph->init();
	dataGenerator->computeNodeDistance(graph);
	forcefield->centeraxis_position = dataGenerator->getZeroPotential();
	trainingModal->setPotentialCenter(dataGenerator->getZeroPotential());
	std::cout << "zero potential" << std::endl;
	std::cout << dataGenerator->getZeroPotential().transpose() << std::endl;

	delete graph;

	/*if (cubeVolumtricMesh_distace != NULL)
	{
	LoboVolumetricMeshGraph* graph = new LoboVolumetricMeshGraph(cubeVolumtricMesh_distace);
	graph->init();
	dataGenerator->computeNodeDistance(graph);
	delete graph;
	}*/
}

void DeepWarpSimulator::doGenerateTrainingDataModel()
{
	if (dataGenerator == NULL)
	{
		std::cout << "dataGenerator not init" << std::endl;
		return;
	}

	invertibleStvkModel->setGravity(false);
	isotropicHyperelasticModel->setAddGravity(false);

	if (!trainingModal->getAlreadyReadData())
	trainingModal->excute();


	updateVolMeshDis(0);
	invertibleStvkModel->setGravity(getUseGravity());
	isotropicHyperelasticModel->setAddGravity(getUseGravity());

	//pause generate
	dataGenerator->setSeperateGenerate(getSeperateGenerateData());
	dataGenerator->setExportNodetype(nodeNeighborTypeExport);
	dataGenerator->generateData();
}

void DeepWarpSimulator::doGenerateDataTestedByDNN()
{
	if (dataGenerator == NULL)
	{
		std::cout << "dataGenerator not init" << std::endl;
		return;
	}

	invertibleStvkModel->setGravity(false);
	isotropicHyperelasticModel->setAddGravity(false);

	if (!trainingModal->getAlreadyReadData())
		trainingModal->excute();

	VectorXd testforce = ((ModalWarpingTrainingModel*)trainingModal)->getNonlinearInternalforce(123);
	std::ofstream test("extforce.txt");
	test << testforce.rows() << std::endl;
	test << testforce << std::endl;
	test.close();

	updateVolMeshDis(0);
	invertibleStvkModel->setGravity(getUseGravity());
	isotropicHyperelasticModel->setAddGravity(getUseGravity());


	//pause generate
	dataGenerator->setSeperateGenerate(getSeperateGenerateData());
	dataGenerator->setExportNodetype(nodeNeighborTypeExport);
	int type = this->getNodeNeighborTypeExport();

	dataGenerator->testDataByDNN(loboNeuralNetwork);

}

void DeepWarpSimulator::searchClosestTrainingData(double threshold)
{
	std::ifstream intest("selectedNodeinput.txt");
	for (int i = 0; i < selectedNodeInput.size(); i++)
	{
		intest >> selectedNodeInput[i];
	}
	intest.close();
	int dimension = dataGenerator->getInputDimension();
	int testDimension = dataGenerator->getTestDimension();
	int numTrainingDis = trainingModal->getNumTrainingSet();

	std::vector<int> poseIndexList;
	for (int i = 0; i < numTrainingDis; i++)
	{
		int disIndex = i;
		VectorXd lq = trainingModal->getTrainingLinearDis(disIndex);
		VectorXd nq = trainingModal->getTrainingNonLinearDis(disIndex);
		VectorXd dq = nq - lq;

		VectorXd extforce;
		trainingModal->getTraingLinearForce(disIndex, extforce);
		int numVertex = volumetric_mesh->getNumVertices();
		VectorXd p(numVertex);

		VectorXd inputDNN;
		((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, inputDNN, extforce);

		VectorXd selectedNodeInputSub(testDimension);
		for (int j = 0; j < testDimension; j++)
		{
			selectedNodeInputSub.data()[j] = selectedNodeInput.data()[j];
		}
		selectedNodes.clear();
		for (int i = 0; i < numVertex; i++)
		{
			VectorXd nodeDNN(testDimension);
			for (int j = 0; j < testDimension; j++)
			{
				nodeDNN.data()[j] = inputDNN.data()[i * dimension + j];
			}

			/*if (p.data()[i] < 0.05)
			{
			selectedNodes.push_back(i);
			}*/

			p.data()[i] = (nodeDNN - selectedNodeInputSub).norm() / selectedNodeInputSub.norm();

			if (p.data()[i] < threshold)
			{
				poseIndexList.push_back(disIndex);
				std::cout << disIndex << std::endl;
				std::cout<<trainingModal->getForceFieldDirection(disIndex).transpose() << std::endl;
				break;
			}
		}
	}
}

void DeepWarpSimulator::updateVolMeshDis(int disIndex, bool nonlinear /*= false*/)
{
	if (trainingModal == NULL)
	{
		return;
	}

	if (!nonlinear)
	{
		VectorXd dis = trainingModal->getTrainingLinearDis(disIndex);
		updateTriAndVolumeMesh(dis.data());
	}
	else
	{
		VectorXd dis = trainingModal->getTrainingNonLinearDis(disIndex);
		updateTriAndVolumeMesh(dis.data());
	}

	VectorXd lq = trainingModal->getTrainingLinearDis(disIndex);
	VectorXd nq = trainingModal->getTrainingNonLinearDis(disIndex);
	VectorXd dq = nq - lq;
	Eigen::AngleAxisd aa(0, Vector3d(-1, 0, 0));
	Matrix3d rotation = aa.toRotationMatrix();

	VectorXd extforce;
	trainingModal->getTraingLinearForce(disIndex,extforce);
	int numVertex = volumetric_mesh->getNumVertices();

	for (int i = 0; i < numVertex; i++)
	{
		Vector3d forcenode;
		forcenode.data()[0] = extforce.data()[i * 3 + 0];
		forcenode.data()[1] = extforce.data()[i * 3 + 1];
		forcenode.data()[2] = extforce.data()[i * 3 + 2];
		forcenode = rotation*forcenode;
		extforce.data()[i * 3 + 0] = forcenode.data()[0];
		extforce.data()[i * 3 + 1] = forcenode.data()[1];
		extforce.data()[i * 3 + 2] = forcenode.data()[2];
	}

	VectorXd p(numVertex);
	//trainingModal->getTraingLinearForce(lq, extforce,poisson);

	//VectorXd w = *(modalrotationMatrix->w_operator)*lq;

	//modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	//extforce = *localOrientationMatrixR*extforce;
	////extforce = invertibleStvkModel->getGravity();

	//((RotationStrainDataGenerator*)dataGenerator)->updatePotentialOrder(extforce, -1);

	//VectorXd p = ((RotationStrainDataGenerator*)dataGenerator)->getNodePotentialE();


	VectorXd inputDNN;
	((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, inputDNN, extforce);
	
	//
	std::ifstream intest("selectedNodeinput.txt");
	for (int i = 0; i < selectedNodeInput.size(); i++)
	{
		intest >> selectedNodeInput[i];
	}
	intest.close();
	int dimension = dataGenerator->getInputDimension();
	int testDimension = dataGenerator->getTestDimension();
	VectorXd selectedNodeInputSub(testDimension);
	for (int j = 0; j < testDimension; j++)
	{
		selectedNodeInputSub.data()[j] = selectedNodeInput.data()[j];
	}

	selectedNodes.clear();

	std::vector<int> nodescaled;
	double errorthreshold = 0.2;
	for (int i = 0; i < numVertex; i++)
	{
		VectorXd nodeDNN(testDimension);
		for (int j = 0; j < testDimension; j++)
		{
			nodeDNN.data()[j] = inputDNN.data()[i * dimension + j];
		}

		/*if (p.data()[i] < 0.05)
		{
			selectedNodes.push_back(i);
		}*/

		p.data()[i] = (nodeDNN - selectedNodeInputSub).norm() / selectedNodeInputSub.norm();
		
		if (p.data()[i] > errorthreshold)
		{
			p.data()[i] = 1;
		}
		else
		{
			//p.data()[i] /= 0.2;
			nodescaled.push_back(i);
		}
	}

	double minValue = DBL_MAX;
	for (int i = 0; i < nodescaled.size(); i++)
	{
		int nodeid =  nodescaled[i];
		if (p.data()[nodeid] < minValue)
		{
			minValue = p.data()[nodeid];
		}
	}

	for (int i = 0; i < nodescaled.size(); i++)
	{
		int nodeid = nodescaled[i];
		p.data()[nodeid] = errorthreshold - (errorthreshold / (errorthreshold - minValue))*(errorthreshold - p.data()[nodeid]);
		p.data()[nodeid] /= errorthreshold;
	}

	volumetric_obj_adapter->setTriMeshVertexColorData1D(p.data());

	
	VectorXd q(numVertex * 3);
	q.setZero();
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d nodep = volumetric_mesh->getNodePosition(i);
		nodep = rotation*nodep;
		Vector3d nodeop = volumetric_mesh->getNodeRestPosition(i);
		for (int j = 0; j < 3; j++)
		{
			q.data()[i * 3 + j] = nodep.data()[j] - nodeop.data()[j];
		}
	}
	updateTriAndVolumeMesh(q.data());
}

void DeepWarpSimulator::saveTrainingData(const char* filename)
{
	if (trainingModal!=NULL)
	trainingModal->saveTrainingSet(filename);
}

void DeepWarpSimulator::readTrainingData(const char* filename)
{
	if (trainingModal != NULL)
	trainingModal->readTrainingSet(filename);
}

void DeepWarpSimulator::getplotTrainingData(std::vector<double> &x, std::vector<double> &y)
{
	
	if (trainingModal == NULL)
	{
		return;
	}

	int numPose = trainingModal->getNumTrainingSet();

	x.clear();
	y.clear();

	for (int i = 0; i < numPose; i += 20)
	{
		VectorXd lq = trainingModal->getTrainingLinearDis(i);
		VectorXd f;
		double poisson = trainingModal->getPoissonPerDis(i);

		int numVertex = lq.rows()/3;
		
		Vector3d fi;
		trainingModal->getTrainingLinearForceDirection(i, fi);
		f.resize(numVertex * 3);

		for (int j = 0; j < numVertex; j++)
		{
			f.data()[j * 3 + 0] = fi.data()[0];
			f.data()[j * 3 + 1] = fi.data()[1];
			f.data()[j * 3 + 2] = fi.data()[2];
		}

		VectorXd dnnInput;
		((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, dnnInput, f);

		int dimension = dataGenerator->getInputDimension();

		for (int j = 0; j < numVertex; j+=20)
		{
			x.push_back(dnnInput.data()[j*dimension + 13-6]);

			y.push_back(dnnInput.data()[j*dimension + 15-6]);
		}
	}

}

void DeepWarpSimulator::loadTrainingDataFeature(std::vector<double> &x, std::vector<double> &y, const char* filename)
{
	if (dataGenerator == NULL)
	{
		std::cout << "you should init simulator first." << std::endl;
		return;
	}

	x.clear();
	y.clear();

	int dimension = dataGenerator->getInputDimension();

	//cnpy::NpyArray arr = cnpy::npy_load(filename);
	//double* loadeddata = arr.data<double>();

	//int datasize = arr.shape[0];
	//
	//if (dimension != arr.shape[1])
	//{
	//	std::cout << "load wrong file." << std::endl;
	//	return;
	//}

	//assert(dimension == arr.shape[1]);

	for (int i = 0; i < dataset.rows(); i += 50)
	{
		x.push_back(dataset.data()[(5) *dataset.rows() + i]);
		y.push_back(dataset.data()[(6) * dataset.rows() + i]);
	}
}

void DeepWarpSimulator::loadTrainingDataPCAFeature(std::vector<double> &x, std::vector<double>&y)
{
	if (dataGenerator == NULL)
	{
		std::cout << "you should init simulator first." << std::endl;
		return;
	}

	x.clear();
	y.clear();

	for (int i = 0; i < outputpca.rows(); i++)
	{
		x.push_back(outputpca.data()[0 * outputpca.rows() + i]);
		y.push_back(outputpca.data()[1 * outputpca.rows() + i]);
	}
}

void DeepWarpSimulator::loadTrainingDataPCA(const char* filename)
{
	int dimension = dataGenerator->getInputDimension();
	cnpy::NpyArray arr = cnpy::npy_load(filename);
	double* loadeddata = arr.data<double>();
	if (dimension != arr.shape[1])
	{
		std::cout << "load wrong file." << std::endl;
		return;
	}

	int rows = arr.shape[0];
	dataset.resize(rows, dimension);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			dataset.data()[j*rows + i] = loadeddata[i*dimension+j];
		}
	}

	PCAMatrix(dataset, outputpca, transformpca, 2, scalingFactors, minCoeffs, featureMeans
	);

	//std::ofstream test2("test2.txt");
	//test2 << transformpca << std::endl;
	//test2.close();

}

void DeepWarpSimulator::getSelectedNodesFeature(std::vector<double> &x, std::vector<double> &y)
{
	VectorXd lq = integrator->getVectorq();
	VectorXd f = invertibleStvkModel->getGravity();
	VectorXd dnnInput;
	((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, dnnInput, f);
	int dimension = dataGenerator->getInputDimension();

	for (int i = 0; i < selectedNodes.size(); i++)
	{
		int nodeid = selectedNodes[i];

		x.push_back(dnnInput.data()[nodeid*dimension + 5]);
		y.push_back(dnnInput.data()[nodeid*dimension + 6]);
	}
}

void DeepWarpSimulator::getSelectedNodesPCAFeature(std::vector<double> &x, std::vector<double> &y)
{
	VectorXd lq = integrator->getVectorq();
	VectorXd f = invertibleStvkModel->getGravity();
	VectorXd dnnInput;
	((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, dnnInput, f);
	int dimension = dataGenerator->getInputDimension();

	for (int i = 0; i < selectedNodes.size(); i++)
	{
		int nodeid = selectedNodes[i];

		RowVectorXd nodefeature(dimension);;
		for (int j = 0; j < dimension; j++)
		{
			nodefeature[j] = dnnInput.data()[nodeid*dimension + j];
		}
		nodefeature = (nodefeature.rowwise() - minCoeffs).array().rowwise() / scalingFactors.array();
		nodefeature = nodefeature.rowwise() - featureMeans;

		VectorXd pcafeature = nodefeature*transformpca;

		x.push_back(pcafeature.data()[0]);
		y.push_back(pcafeature.data()[1]);
	}
}

void DeepWarpSimulator::getNodePotentialValue()
{
	VectorXd lq = integrator->getVectorq();

	int numVertex = volumetric_mesh->getNumVertices();
	VectorXd p(numVertex);
	//trainingModal->getTraingLinearForce(lq, extforce,poisson);

	//VectorXd w = *(modalrotationMatrix->w_operator)*lq;

	//modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	//extforce = *localOrientationMatrixR*extforce;
	////extforce = invertibleStvkModel->getGravity();

	//((RotationStrainDataGenerator*)dataGenerator)->updatePotentialOrder(extforce, -1);

	//VectorXd p = ((RotationStrainDataGenerator*)dataGenerator)->getNodePotentialE();

	VectorXd inputDNN;
	((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, inputDNN, invertibleStvkModel->getGravity());

	//((RotationTrainingDataGeneratorV4*)dataGenerator)->updatePotentialOrder(invertibleStvkModel->getGravity(),-1);

	//VectorXd p = ((RotationTrainingDataGeneratorV4*)dataGenerator)->getNodeDistance();
	//volumetric_obj_adapter->setTriMeshVertexColorData1D(p.data());
	int dimension = dataGenerator->getInputDimension();
	for (int i = 0; i < numVertex; i++)
	{
		//p.data()[i] = (inputDNN.data()[i*dimension + dimension - 3]);

		p.data()[i] = (inputDNN.data()[i*dimension + dimension - 1]/2.0)+0.5;
	}

	//volumetric_obj_adapter->setColorMap(Vector3d(1, 1, 1), Vector3d(0.1, 0.1, 1));
	volumetric_obj_adapter->setTriMeshVertexColorData1D(p.data());
	//volumetric_obj_adapter->exportVertexColorMesh1D("barmap.obj", p.data());
}

void DeepWarpSimulator::exportFeatureMap(int featureid)
{
	VectorXd lq = integrator->getVectorq();
	int numVertex = volumetric_mesh->getNumVertices();
	VectorXd p(numVertex);
	//trainingModal->getTraingLinearForce(lq, extforce,poisson);

	//VectorXd w = *(modalrotationMatrix->w_operator)*lq;

	//modalrotationMatrix->computeLocalOrientationMatrixR(localOrientationMatrixR, w, false);
	//extforce = *localOrientationMatrixR*extforce;
	////extforce = invertibleStvkModel->getGravity();

	//((RotationStrainDataGenerator*)dataGenerator)->updatePotentialOrder(extforce, -1);

	//VectorXd p = ((RotationStrainDataGenerator*)dataGenerator)->getNodePotentialE();

	VectorXd inputDNN;

	if (((RotationStrainDataGenerator*)dataGenerator)->getInputtype() == 0)
	{
		((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, inputDNN, invertibleStvkModel->getGravity());
	}
	else
	{
		VectorXd forceaxis(numVertex * 3);
		forceaxis.setZero();
		forceaxis.data()[0] = forcefield->centeraxis.data()[0];
		forceaxis.data()[1] = forcefield->centeraxis.data()[1];
		forceaxis.data()[2] = forcefield->centeraxis.data()[2];
		((RotationStrainDataGenerator*)dataGenerator)->createInputData(lq, inputDNN, forceaxis);
	}
	//((RotationTrainingDataGeneratorV4*)dataGenerator)->updatePotentialOrder(invertibleStvkModel->getGravity(),-1);

	//VectorXd p = ((RotationTrainingDataGeneratorV4*)dataGenerator)->getNodeDistance();
	//volumetric_obj_adapter->setTriMeshVertexColorData1D(p.data());
	int dimension = dataGenerator->getInputDimension();
	for (int i = 0; i < numVertex; i++)
	{
		//p.data()[i] = (inputDNN.data()[i*dimension + dimension - 3]);
		if (featureid == 2)
		p.data()[i] = (inputDNN.data()[i*dimension + dimension - 1] / 2.0) + 0.5;

		if (featureid == 1)
			p.data()[i] = (inputDNN.data()[i*dimension + dimension - 2] );

		if (featureid == 0)
			p.data()[i] = (inputDNN.data()[i*dimension + dimension-3]);
	}

	//volumetric_obj_adapter->setColorMap(Vector3d(1, 1, 1), Vector3d(0.1, 0.1, 1));
	volumetric_obj_adapter->setTriMeshVertexColorData1D(p.data());
	volumetric_obj_adapter->exportVertexColorMesh1D("barmap.obj", p.data());
}

void DeepWarpSimulator::generateLinearRotationTrainingData()
{
	std::vector<Vector3d> axisangle;
	std::vector<Vector3d> potentialDirection;
	std::vector<bool> ifreset;
	std::vector<Matrix3d> Transform_;
	std::vector<Matrix3d> Transform_n;

	generateRotationVectorSample(M_PI*2, 0, 20,
		M_PI/2.0, -M_PI / 2.0, 20,
		M_PI/2, 0.1, 10, axisangle, ifreset, potentialDirection);

	for (int i = 0; i < axisangle.size(); i++)
	{
		Matrix3d R_L;
		skewMatrix(axisangle[i], R_L);
		R_L += Matrix3d::Identity();
		Transform_.push_back(R_L);

		Matrix3d R_N;
		AngleAxis<double> aa(axisangle[i].norm(), axisangle[i].normalized());
		R_N = aa.toRotationMatrix();
		Transform_n.push_back(R_N);
	}

	int numVertex = volumetric_mesh->getNumVertices();
	
	VectorXd q(numVertex*3);
	q.setZero();
	
	//ground true outputMatrix
	std::vector<double> input_data;
	std::vector<double> output_data;
	std::vector<double> input_test;
	std::vector<double> output_test;


	for (int i = 0; i < axisangle.size(); i++)
	{
		for (int j = 0; j < numVertex; j++)
		{
			Vector3d nodep = volumetric_mesh->getNodeRestPosition(j);
			Vector3d nodecp = Transform_n[i] * nodep;
			Vector3d nq = nodecp - nodep;
			
			//linear shape
			nodecp = Transform_[i] * nodep;
			Vector3d lq = nodecp - nodep;

			if ( i%2 == 0)
			{
				output_data.push_back(lq.data()[0]);
				output_data.push_back(lq.data()[1]);
				output_data.push_back(lq.data()[2]);
			}
			else
			{
				output_test.push_back(lq.data()[0]);
				output_test.push_back(lq.data()[1]);
				output_test.push_back(lq.data()[2]);
			}
		
			
			q.data()[j * 3 + 0] = lq.data()[0];
			q.data()[j * 3 + 1] = lq.data()[1];
			q.data()[j * 3 + 2] = lq.data()[2];
		}

		volumetric_mesh->setDisplacement(q.data());
		VectorXd w = *modalRotationSparseMatrix*q;
		Matrix3d R_polar;
		for (int j = 0; j < numVertex; j++)
		{
			Matrix3d R_j;
			volumetric_mesh->computeNodeRotationRing(j, R_j);
			AngleAxisd aa;
			aa = R_j;

			Vector3d axis = aa.axis();
			double angle = aa.angle();

			/*Vector3d nodew;
			nodew.data()[0] = w.data()[j * 3 + 0];
			nodew.data()[1] = w.data()[j * 3 + 1];
			nodew.data()[2] = w.data()[j * 3 + 2];
			axis = nodew.normalized();
			angle = nodew.norm();*/

			Vector3d nodeq = volumetric_mesh->getNodeRestPosition(j);


			if (i % 2 == 0)
			{
				for (int k = 0; k < 3; k++)
				{
					input_data.push_back(axis.data()[k]);
				}
				input_data.push_back(angle);
				for (int k = 0; k < 3; k++)
				{
					input_data.push_back(nodeq.data()[k]);
				}
			}
			else
			{
				for (int k = 0; k < 3; k++)
				{
					input_test.push_back(axis.data()[k]);
				}
				input_test.push_back(angle);
				for (int k = 0; k < 3; k++)
				{
					input_test.push_back(nodeq.data()[k]);
				}
			}
		}
	}

	int dimension = 3+1+3;
	int outputdimension = 3;
	std::ostringstream data[4];
	std::string elefilename[4];
	data[0] << "input_train"  << ".txt";
	elefilename[0] = data[0].str();

	data[1] << "output_train" << ".txt";
	elefilename[1] = data[1].str();

	data[2] << "input_test"  << ".txt";
	elefilename[2] = data[2].str();

	data[3] << "output_test"  << ".txt";
	elefilename[3] = data[3].str();

	trainingModal->exportExampleData(elefilename[0].c_str(), input_data.data(), dimension, input_data.size());
	trainingModal->exportExampleData(elefilename[1].c_str(), output_data.data(), outputdimension, output_data.size());

	trainingModal->exportExampleData(elefilename[2].c_str(), input_test.data(), dimension, input_test.size());
	trainingModal->exportExampleData(elefilename[3].c_str(), output_test.data(), outputdimension, output_test.size());
	std::cout << "datafinished" << std::endl;
}

void DeepWarpSimulator::generateEnergyCurveData()
{
	int numdata = 100;
	int numVertex = volumetric_mesh->getNumVertices();
	VectorXd q(numVertex * 3);
	std::vector<double> x;
	std::vector<double> y;
	for (int i = 0; i < numdata; i++)
	{
		double scale = (i / 100.0)*2.0+1.0;

		q.setZero();
		for (int j = 0; j < numVertex; j++)
		{
			Vector3d nodep = volumetric_mesh->getNodeRestPosition(j);
			q.data()[j * 3 + 0] = (nodep.data()[0]-0.5) * scale +0.5 - nodep.data()[0];
		}

		volumetric_mesh->setDisplacementBuffer(q.data());

		double energy = hyperforcemodel->computeElementEnergy(0);

		/*int numElements = volumetric_mesh->getNumElements();
		energy *= numElements;*/

		x.push_back(scale-1);
		y.push_back(energy);
	}

	std::ofstream outputStream("stressmap.txt");
	for (int i = 0; i < x.size(); i++)
	{
		outputStream << x[i] << " " << y[i] << std::endl;
	}
	outputStream.close();
}

void DeepWarpSimulator::setGenLinearShapeOnly(bool b)
{
	if (trainingModal == NULL)
	{
		return;
	}
	trainingModal->setLinearDisOnly(b);
}

void DeepWarpSimulator::initIntegrator()
{
	int r = volumetric_mesh->getNumVertices() * 3;

	/*integrator = new ImplicitNewMatkSparseIntegrator(r, timestep, massMatrix, stvkforce_model, 1, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef, newtonIteration);*/

	/*integrator = new ImplicitNewMarkDNNIntegrator(modalrotationMatrix, modalRotationSparseMatrix, DNNpath.c_str(), r, timestep, massMatrix, stvkforce_model, 1, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef, newtonIteration, 1e-6, 0.25, 0.5,false);*/

	integrator = new ImplicitModalWarpingIntegrator(modalrotationMatrix, modalRotationSparseMatrix, r, timestep, massMatrix, stvkforce_model, 1, numConstrainedDOFs, constrainedDOFs, dampingMassCoef, dampingStiffnessCoef, newtonIteration, 1e-6, 0.25, 0.5, false);


	std::cout << "integrator_skip" << integrator_skip << std::endl;
	integrator->setSkipSteps(integrator_skip);
	std::cout << "finished init integrator" << std::endl;
}

void DeepWarpSimulator::readConfig(std::ifstream &inStream)
{
	std::string token;
	while (true)
	{
		inStream >> token;

		if (inStream.eof())
		{
			break;
		}

		if (token == "end")
		{
			std::cout << "simulator config finished ..." << std::endl;
			std::cout << std::endl;
			break;
		}

		if (token[0] == '#')
		{
			inStream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}

		if (token == "load")
		{
			if (!ownVolumetric_mesh)
			{
				std::cout << "This simulator will use exist volumetric mesh." << std::endl;
				continue;
			}

			std::cout << "load volumetric mesh ...";
			std::string filebase;
			inStream >> filebase;
			std::cout << filebase << std::endl;
			this->readVolumetricMeshAscii(filebase.c_str());
		}

		if (token == "loadcube")
		{
			std::cout << "load cube mesh ..." << std::endl;
			std::string filebase;
			inStream >> filebase;
			std::cout << filebase << std::endl;
			this->readCubeMeshAscii(filebase.c_str());
		}

		if (token == "loadCubeFordistance")
		{
			std::cout << "load cube mesh ..." << std::endl;
			std::string filebase;
			inStream >> filebase;
			this->creadCubeMeshDistanceAscii(filebase.c_str());
		}

		if (token == "constraints")
		{
			std::string filebase;
			inStream >> filebase;
			this->readConstraints(filebase.c_str());
		}

		if (token == "DNNpath")
		{
			inStream >> DNNpath;
		}

		if (token == "nodeNeighborTypeExport")
		{
			inStream >> nodeNeighborTypeExport;
		}

		if (token == "seperateGenerateData")
		{
			bool b;
			inStream >> b;
			setSeperateGenerateData(b);
		}
	}
}

void DeepWarpSimulator::mergeExternalForce()
{
	invertibleStvkModel->setGravity(false);

	int numVertex = volumetric_mesh->getNumVertices();

	//int R = numVertex * 3;
	//VectorXd fullq(R);
	//memcpy(fullq.data(), integrator->getq(), R*sizeof(double));
	//VectorXd w = (*modalRotationSparseMatrix)*fullq;
	//modalrotationMatrix->computeWarpingRotationMatrixR(rotationSparseMatrixR, w);
	//fullq = (*rotationSparseMatrixR)*fullq;
	//volumetric_mesh->setDisplacement(fullq.data());
	
	//forcefield->computeCurExternalForce(externalForce);
	mergedExternalForce = externalForce + mouseForce + collisionExternalForce;

	mergedExternalForce += invertibleStvkModel->getGravity();

	integrator->setExternalForces(mergedExternalForce.data());
}	

void DeepWarpSimulator::exportFeatureData(const char* filename, std::vector<double> features, int dimension)
{
	std::ofstream outputstream(filename);

	int num_set = features.size() / dimension;
	for (int i = 0; i < num_set; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			outputstream << features.data()[i * dimension + j] << ",";
		}
		outputstream << std::endl;
	}
	outputstream.close();
}

void DeepWarpSimulator::exportTargetData(const char* filename, std::vector<double> targets, int dimension)
{
	std::ofstream outputstream(filename);
	int num_set = targets.size() / dimension;
	for (int i = 0; i < num_set; i++)
	{
		for (int j = 0; j < dimension; j++)
		{
			outputstream << targets.data()[i * dimension + j] << ",";
		}
		outputstream << std::endl;
	}
	outputstream.close();
}

DeepWarpSimulator* downCastModalwarpingSimulator(LoboSimulatorBase* simulator)
{
	return (DeepWarpSimulator*)simulator;
}
