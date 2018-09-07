#pragma once
#include <vector>

class CubeVolumtricMesh;
class TetVolumetricMesh;
class CubeNodeMapType;

class TeMeshVoxlizeModel
{
public:
	TeMeshVoxlizeModel(CubeVolumtricMesh* cubeMesh_,TetVolumetricMesh* tetMesh_);
	~TeMeshVoxlizeModel();

	virtual void createCubeTetMapping();


	//************************************
	// Method:    getNodeCube
	// FullName:  TeMeshVoxlizeModel::getNodeCube
	// Access:    public 
	// Returns:   cube id
	// Qualifier:
	// Parameter: int nodeid
	//************************************
	virtual int getNodeCube(int nodeid);
	virtual int getCubeNode(int cubeid, int nodeindex);
	virtual int getCubeNodeSize(int cubeid);
	virtual CubeNodeMapType* getCubeNodeMapRefer(int nodeid);


protected:

	std::vector<CubeNodeMapType*> cubenodeMaps_list;
	std::vector<int> tetNode_Cube;
	std::vector<std::vector<int>> cube_TetNode;


	CubeVolumtricMesh* cubeMesh;
	TetVolumetricMesh* tetMesh;

};

