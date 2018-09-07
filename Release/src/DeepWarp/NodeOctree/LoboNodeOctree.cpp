#include "LoboNodeOctree.h"
#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"

LoboNodeOctree::LoboNodeOctree(LoboVolumetricMesh* volumtricMesh_, LoboVolumetricMeshGraph* volumtricMeshGraph_, double voxSize_, int treeDepth_)
{
	this->volumtricMesh = volumtricMesh_;
	this->volumtricMeshGraph = volumtricMeshGraph_;
	this->voxSize = voxSize_;
	this->treeDepth = treeDepth_;
}

LoboNodeOctree::~LoboNodeOctree()
{
}

void LoboNodeOctree::init()
{
	int numVertex = volumtricMesh->getNumVertices();
	node_Octree_root.resize(numVertex);
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d node_position = volumtricMesh->getNodePosition(i);
		node_Octree_root[i] = new LoboOctant(node_position, voxSize, 2, NULL);
		node_Octree_root[i]->subdivide(volumtricMesh, volumtricMeshGraph);
	}
}

LoboOctant::LoboOctant(Vector3d center_, double length_, int depth_, LoboOctant* parent_)
{
	if (depth_ == 0)
	{
		leef = true;
	}
	else
	{
		leef = false;
	}
	empty = true;

	this->center = center_;
	this->length = length_;
	this->depth = depth_;
	this->parent = parent_;

}

LoboOctant::~LoboOctant()
{
	
}

bool LoboOctant::ifInOctant(Vector3d position)
{
	double distance = (position - center).lpNorm<1>();
	if (distance < this->length / 2.0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void LoboOctant::subdivide(LoboVolumetricMesh* volumtricMesh_, LoboVolumetricMeshGraph* volumtricMeshGraph_)
{
	//create 8 children
	for (int i = 0; i < 8; i++)
	{
		int c[3];
		c[0] = i % 2;
		c[1] = i % 4 / 2;
		c[2] = i / 4;

		Vector3d childeCenter;
		double sublen = length / 2.0;
		for (int j = 0; j < 3;j++)
		childeCenter.data()[j] = center.data()[j] + (c[j] * 2 - 1)*sublen/2.0;
		double subdepth = depth - 1;
		LoboOctant* octanct = new LoboOctant(childeCenter, sublen, subdepth, this);
		this->children.push_back(octanct);
	}

	if (parent == NULL)
	{
		//this node is root, we need find the node index first
		int numVertex = volumtricMesh_->getNumVertices();
		for (int i = 0; i < numVertex; i++)
		{
			Vector3d nodeposition = volumtricMesh_->getNodePosition(i);
			if (ifInOctant(nodeposition))
			{
				nodeindex.push_back(i);
			}
		}
	}

	if (parent != NULL)
	{
		for (int i = 0; i < this->nodeindex.size(); i++)
		{
			int node_index = this->nodeindex[i];
			Vector3d nodeposition = volumtricMesh_->getNodePosition(node_index);

			for (int j = 0; j < children.size(); j++)
			{
				if (children[j]->ifInOctant(nodeposition))
				{
					children[j]->nodeindex.push_back(node_index);
					continue;
				}
			}
		}
	}


	//check if the octant is empty and continue subdivide
	for (int i = 0; i < children.size(); i++)
	{
		if (children[i]->nodeindex.size() == 0)
		{
			children[i]->empty = true;
		}
		else
		{
			if (!children[i]->leef)
			{
				children[i]->subdivide(volumtricMesh_,volumtricMeshGraph_);
			}
		}
	}

}
