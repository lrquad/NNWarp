#pragma once
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

class LoboNodeOctree;

class LoboNodeOctreeRender
{
public:
	LoboNodeOctreeRender(LoboNodeOctree* loboNodeOctree);
	~LoboNodeOctreeRender();

	virtual void renderOctree(QOpenGLShaderProgram *program);

protected:

	virtual void drawVoxel(QOpenGLShaderProgram *program,LoboNodeOctree* nodeOctant);

	LoboNodeOctree* nodeOctree;

};

