#include "LoboNodeOctreeRender.h"
#include "LoboNodeOctree.h"



LoboNodeOctreeRender::LoboNodeOctreeRender(LoboNodeOctree* loboNodeOctree_)
{
	this->nodeOctree = loboNodeOctree_;
}

LoboNodeOctreeRender::~LoboNodeOctreeRender()
{

}

void LoboNodeOctreeRender::renderOctree(QOpenGLShaderProgram *program)
{

}

void LoboNodeOctreeRender::drawVoxel(QOpenGLShaderProgram *program, LoboNodeOctree* nodeOctant)
{
	glBegin(GL_QUADS);




	glEnd();
}
