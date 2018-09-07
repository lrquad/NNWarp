#pragma once
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

class LoboVolumetricMesh;
class LoboVolumetricMeshGraph;
class LoboOctant;


class LoboNodeOctree
{
public:
	LoboNodeOctree(LoboVolumetricMesh* volumtricMesh_, LoboVolumetricMeshGraph* volumtricMeshGraph_, double voxSize_, int treeDepth_);
	~LoboNodeOctree();

	virtual void init();

	double voxSize;
	int treeDepth;

	LoboVolumetricMesh* volumtricMesh;
	LoboVolumetricMeshGraph* volumtricMeshGraph;

	std::vector<LoboOctant*> node_Octree_root;
};

class LoboOctant
{
public:

	LoboOctant(Vector3d center,double length_, int depth_, LoboOctant* parent_);
	~LoboOctant();

	bool ifInOctant(Vector3d position);
	void subdivide(LoboVolumetricMesh* volumtricMesh_, LoboVolumetricMeshGraph* volumtricMeshGraph_);

	LoboOctant* parent;
	std::vector<LoboOctant*> children;
	std::vector<int> nodeindex;

	double length;
	Vector3d center;

	int depth;
	bool leef;
	bool empty;

};