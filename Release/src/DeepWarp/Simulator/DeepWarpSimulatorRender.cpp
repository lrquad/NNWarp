#include "DeepWarpSimulatorRender.h"
#include "DeepWarpSimulator.h"

#include "LoboVolumetricMesh/LoboVolumetriceMeshCore.h"
#include "LoboVolumetricMesh/VolumetricMeshRender.h"
#include "LoboVolumetricMesh/TetVolumetricMeshRender.h"
#include "LoboVolumetricMesh/CubeVolumtricMeshRender.h"
#include <iomanip>


DeepWarpSimulatorRender::DeepWarpSimulatorRender(DeepWarpSimulator* simulator) :SimulatorRenderBase(simulator)
{
	this->modalwarpingSimulator = simulator;
	tetvolumetricRender = new TetVolumetricMeshRender();
	cubevolumetricRender = new CubeVolumtricMeshRender();
	mesh_color_index = 1;
}

DeepWarpSimulatorRender::~DeepWarpSimulatorRender()
{
	delete tetvolumetricRender;
	delete cubevolumetricRender;
	delete cubevolumetricRender;
}

void DeepWarpSimulatorRender::renderSimulator(QOpenGLShaderProgram *program)
{
	if (renderConstrained)
	{
		//drawMeshConstrainedNodes(program);
	}

	if (renderselected)
	{
		drawMeshSelected(program);
	}

	
	if (getRender_volume_mesh())
		//tetvolumetricRender->renderAllVolumetricMesh(program, downCastTetVolMesh(simulator->getVolumetricMesh()), mesh_color_index);
		tetvolumetricRender->renderSurfaceVolumstricMesh(program, downCastTetVolMesh(simulator->getVolumetricMesh()), mesh_color_index);

	drawForce(program);
	//drawAllExForce(program);

	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	//glEnable(GL_BLEND); //Enable blending.
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//cubevolumetricRender->renderAllVolumtricMesh(program, downCastCubeVolMesh(modalwarpingSimulator->getCubeVolumtricMesh()), Vector3d::Zero(),1,4);

	//drawLocalFrame(program,Vector3d(0,1,0));
	//drawLocalFrame(program, Vector3d(1, 0, 0));
	//drawLocalFrame(program, Vector3d(0, 0, 1));
	drawSingleSelected(program);

}

void DeepWarpSimulatorRender::paintSimulator(QPainter &painter)
{
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2) << "Steps: " << simulator->getSimulation_steps();
	std::string var = oss.str();
	painter.drawText(10, 40, var.c_str());
}

void DeepWarpSimulatorRender::saveRenderInfo(const char* filename, fileFormatType filetype) const
{

}

void DeepWarpSimulatorRender::drawAllExForce(QOpenGLShaderProgram* program)
{
	if (simulator->getSimulatorState() == LoboSimulatorBase::beforeinit)
	{
		return;
	}
	int numVertex = simulator->getVolumetricMesh()->getNumVertices();
	VectorXd externalforce = modalwarpingSimulator->getMergedExternalForce();
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d position = simulator->getVolumetricMesh()->getNodePosition(i);
		Vector3d nodeforce;
		nodeforce.data()[0] = externalforce[i * 3 + 0];
		nodeforce.data()[1] = externalforce[i * 3 + 1];
		nodeforce.data()[2] = externalforce[i * 3 + 2];

		Vector3d force = position + 1.0 / simulator->getMouseForceRatio()*nodeforce;

		drawArrow(program, position, force);
	}
}

void DeepWarpSimulatorRender::drawLocalFrame(QOpenGLShaderProgram* program, Vector3d axis)
{
	if (!modalwarpingSimulator->getFinishedInit())
	{
		return;
	}

	VectorXd nodeYAxis;
	modalwarpingSimulator->getNodeOrientation(nodeYAxis,axis);

	int numVertex = simulator->getVolumetricMesh()->getNumVertices();
	for (int i = 0; i < numVertex; i++)
	{
		Vector3d position = simulator->getVolumetricMesh()->getNodePosition(i);
		Vector3d end;
		end.data()[0] = position.data()[0] + nodeYAxis.data()[i * 3 + 0]*0.1;
		end.data()[1] = position.data()[1] + nodeYAxis.data()[i * 3 + 1]*0.1;
		end.data()[2] = position.data()[2] + nodeYAxis.data()[i * 3 + 2]*0.1;

		drawArrow(program, position, end);
	}
}

DeepWarpSimulatorRender* downCastModalWarpingRender(SimulatorRenderBase* render)
{
	return (DeepWarpSimulatorRender*)render;
}
