#pragma once
#include "Simulator/simulatorBase/simulatorRenderBase.h"

class DeepWarpSimulator;
class TetVolumetricMeshRender;
class CubeVolumtricMeshRender;

class DeepWarpSimulatorRender:public SimulatorRenderBase
{
public:
	DeepWarpSimulatorRender(DeepWarpSimulator* simulator);
	~DeepWarpSimulatorRender();

	virtual void renderSimulator(QOpenGLShaderProgram *program);
	virtual void paintSimulator(QPainter &painter);

	virtual void saveRenderInfo(const char* filename, fileFormatType filetype) const; // save all render info like frame's position or picked node position and so on.

protected:

	virtual void drawAllExForce(QOpenGLShaderProgram* program);
	virtual void drawLocalFrame(QOpenGLShaderProgram* program,Vector3d axis);

	TetVolumetricMeshRender* tetvolumetricRender;
	DeepWarpSimulator* modalwarpingSimulator;
	CubeVolumtricMeshRender* cubevolumetricRender;
};

DeepWarpSimulatorRender* downCastModalWarpingRender(SimulatorRenderBase* render);
