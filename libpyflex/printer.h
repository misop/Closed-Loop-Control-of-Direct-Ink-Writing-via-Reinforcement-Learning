#include <NvFlex.h>
#include <NvFlexExt.h>
#include <NvFlexDevice.h>

#include "../core/maths.h"
#include "../core/mesh.h"
#include "helpers.h"

struct Emitter {
	Emitter() : mSpeed(0.0f), mEnabled(false), mLeftOver(0.0f), mWidth(8) {}

	Vec3 mPos;
	Vec3 mDir;
	Vec3 mRight;
	Vec3 mVelocity;
	float mSpeed;
	float mFlow;
	bool mEnabled;
	float mLeftOver;
	int mWidth;
};

class Printer {
public:

	Printer(NvFlexLibrary* g_flexLib, float viscosity = 10.0f, float dissipation = 0.24f) : mflexLib(g_flexLib), viscosity(viscosity), dissipation(dissipation) {}

	Emitter Initialize(SimBuffers *g_buffers, std::map<NvFlexTriangleMeshId, OpenGL::GpuMesh*> &g_meshes, NvFlexParams &g_params, float viscosityAlpha, int materialReservoir, int meshID) {
        viscosity = (1.f - viscosityAlpha) * 0.5 + viscosityAlpha * 10.f;
		dissipation = 0.24 * viscosityAlpha;
		float radius = 0.1f;
		float restDistance = radius*0.5f;

		g_params.radius = radius;
		g_params.numIterations = 3;
		g_params.vorticityConfinement = 0.0f;
		g_params.fluidRestDistance = restDistance;
		g_params.smoothing = 0.35f;
		g_params.relaxationFactor = 1.f;
		g_params.restitution = 0.0f;
		g_params.collisionDistance = restDistance;
		g_params.shapeCollisionMargin = g_params.collisionDistance*0.25f;
		g_params.dissipation = dissipation;

		g_params.dynamicFriction = 1.0f;
		g_params.staticFriction = 0.0f;
		g_params.viscosity = 20.0f + 20.0f*viscosity;
		g_params.adhesion = std::min(0.1f*viscosity, 0.5f);
		g_params.adhesion = 0.f;
		g_params.cohesion = 0.05f*viscosity;//*2.f
		g_params.surfaceTension = 0.0f;
		g_params.gravity[1] *= 2.f;

		const float shapeSize = 2.0f;
		const Vec3 shapeLower = Vec3(-shapeSize*0.5f, 0.0f, -shapeSize*0.5f);
		const Vec3 shapeUpper = shapeLower + Vec3(shapeSize);
		const Vec3 shapeCenter = (shapeLower + shapeUpper)*0.5f;

		Mesh *shape = NULL;
		switch (meshID) {
		case 0:
			shape = ImportMesh("./data/nozzle_cad.obj");
			break;
		case 1:
			shape = ImportMesh("./data/nozzle_cad_thick.obj");
			break;
		case 2:
			shape = ImportMesh("./data/nozzle_cad_subdiv.obj");
			break;
		}
		shape->Transform(ScaleMatrix(Vec3(0.7f)));
		mesh = CreateTriangleMesh(mflexLib, g_meshes, shape);
		AddTriangleMesh(mflexLib, g_buffers, mesh, Vec3(0.f, 0.5f, 0.0f), Quat(), 0.5f);
		delete shape;

		float emitterSize = 0.5f;

		Emitter e;
		e.mEnabled = true;
		e.mWidth = int(emitterSize / restDistance);
		e.mPos = Vec3(0.f, 1.f, 0.f);
		e.mDir = Vec3(0.0f, -1.0f, 0.0f);
		e.mRight = Vec3(1.0f, 0.0f, 0.0f);
		e.mFlow = 0.f;
		e.mSpeed = 1.0f;

		mTime = 0.f;
		vTime = 0.f;

        return e;
	}

	void Update(SimBuffers *g_buffers, Emitter &e, Vec4 updateVector, float g_dt) {
		// update pressure in nozzle
		e.mFlow = updateVector.w;
		// update nozzle position
		ClearShapes(g_buffers);
		// update time buffer
		mTime += g_dt;
		// update emitter
		Vec3 new_location(updateVector);
		Vec3 old_location = e.mPos;
		e.mPos = new_location;
		e.mVelocity = (new_location - old_location) / g_dt;
		// alpha blend the mesh location
		Vec3 posNozzle = new_location;
		posNozzle.y -= 0.5f;
		AddTriangleMesh(mflexLib, g_buffers, mesh, posNozzle, Quat(), 0.5f);

		g_buffers->shapePrevPositions[0] = Vec4(prevPosNozzle, 0.0f);
		g_buffers->shapePrevRotations[0] = Quat();

		// store nozzle location
		prevPosNozzle = posNozzle;
	}

	float viscosity;
	float dissipation;
	float mTime;
	float vTime;
	float mVelocity;

	Vec3 prevPosNozzle;

	NvFlexTriangleMeshId mesh;
    NvFlexLibrary* mflexLib;
};