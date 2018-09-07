#include "TeMeshVoxlizeModel.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "LoboVolumetricMesh/CubeNodeMap.h"

TeMeshVoxlizeModel::TeMeshVoxlizeModel(CubeVolumtricMesh* cubeMesh_, TetVolumetricMesh* tetMesh_)
{
	this->cubeMesh = cubeMesh_;
	this->tetMesh = tetMesh_;
}

TeMeshVoxlizeModel::~TeMeshVoxlizeModel()
{

}

void TeMeshVoxlizeModel::createCubeTetMapping()
{
	int numTetVertex = tetMesh->getNumVertices();
	int numCube = cubeMesh->getNumElements();
	tetNode_Cube.resize(numTetVertex);
	cube_TetNode.resize(numTetVertex);

	for (int i = 0; i < numTetVertex; i++)
	{
		Vector3d position = tetMesh->getNodeRestPosition(i);
		tetNode_Cube[i] = -1;
		for (int j = 0; j < numCube; j++)
		{
			bool isIn = cubeMesh->containsVertex(j, position);
			if (isIn)
			{
				tetNode_Cube[i] = j;
				cube_TetNode[j].push_back(i);
				break;
			}
		}
	}

	cubenodeMaps_list.clear();
	for (int i = 0; i < numTetVertex; i++)
	{
		int cubeid = tetNode_Cube[i];
		CubeElement* element = cubeMesh->getCubeElement(cubeid);
		std::vector<int> neighbors = element->neighbors;
		int count = 0;
		for (int j = 0; j < neighbors.size(); j++)
		{
			if (neighbors[j] != -1)
			{
				count++;
			}
		}

		if (count == 3)
		{
			cubenodeMaps_list.push_back(new CubeNodeThreeNeighborType(neighbors));
		}else
		if (count == 4)
		{
			cubenodeMaps_list.push_back(new CubeNodeFourNeighborType(neighbors));
		}else
		if (count == 5)
		{
			cubenodeMaps_list.push_back(new CubeNodeFiveNeighborType(neighbors));
		}else
		if (count == 6)
		{
			cubenodeMaps_list.push_back(new CubeNodeALLNeighborType(neighbors));
		}
		else
		{
			std::cout << count << std::endl;
			std::cout <<"cubeid"<< cubeid << std::endl;
			std::cout <<"nodeid"<< i << std::endl;
		}

		cubenodeMaps_list.back()->computeSubTypeInfo();
	}

	std::cout << "check node map" << std::endl;
	std::cout << numTetVertex << std::endl;
	std::cout << cubenodeMaps_list.size() << std::endl;
}

int TeMeshVoxlizeModel::getNodeCube(int nodeid)
{
	return tetNode_Cube[nodeid];
}

int TeMeshVoxlizeModel::getCubeNode(int cubeid, int nodeindex)
{
	return cube_TetNode[cubeid][nodeindex];
}

int TeMeshVoxlizeModel::getCubeNodeSize(int cubeid)
{
	return cube_TetNode[cubeid].size();
}

CubeNodeMapType* TeMeshVoxlizeModel::getCubeNodeMapRefer(int nodeid)
{
	return cubenodeMaps_list[nodeid];
}
