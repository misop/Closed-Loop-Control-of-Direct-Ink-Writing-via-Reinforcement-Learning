#include <NvFlex.h>
#include <NvFlexExt.h>
#include <NvFlexDevice.h>
#include <map>

#include "../core/maths.h"

#include "./opengl/shadersGL.h"

struct GpuTimers {
	unsigned long long renderBegin;
	unsigned long long renderEnd;
	unsigned long long renderFreq;
	unsigned long long computeBegin;
	unsigned long long computeEnd;
	unsigned long long computeFreq;
};

OpenGL::GpuMesh* CreateGpuMesh(const Mesh* m) {
	return m ? OpenGL::CreateGpuMesh(m) : nullptr;
}

NvFlexTriangleMeshId CreateTriangleMesh(NvFlexLibrary* g_flexLib, std::map<NvFlexTriangleMeshId, OpenGL::GpuMesh*> &g_meshes, Mesh* m) {
	if (!m)
		return 0;

	Vec3 lower, upper;
	m->GetBounds(lower, upper);

	NvFlexVector<Vec4> positions(g_flexLib, m->m_positions.size());
	positions.map();
	NvFlexVector<int> indices(g_flexLib);

	for (int i = 0; i < int(m->m_positions.size()); ++i) {
		Vec3 vertex = Vec3(m->m_positions[i]);
		positions[i] = Vec4(vertex, 0.0f);
	}
	indices.assign((int*)&m->m_indices[0], m->m_indices.size());

	positions.unmap();
	indices.unmap();

	NvFlexTriangleMeshId flexMesh = NvFlexCreateTriangleMesh(g_flexLib);
	NvFlexUpdateTriangleMesh(g_flexLib, flexMesh, positions.buffer, indices.buffer, m->GetNumVertices(), m->GetNumFaces(), (float*)&lower, (float*)&upper);

	// entry in the collision->render map
	g_meshes[flexMesh] = CreateGpuMesh(m);
	
	return flexMesh;
}

void ClearShapes(SimBuffers *g_buffers) {
	g_buffers->shapeGeometry.resize(0);
	g_buffers->shapePositions.resize(0);
	g_buffers->shapeRotations.resize(0);
	g_buffers->shapePrevPositions.resize(0);
	g_buffers->shapePrevRotations.resize(0);
	g_buffers->shapeFlags.resize(0);
}

void AddTriangleMesh(NvFlexLibrary* g_flexLib, SimBuffers *g_buffers, NvFlexTriangleMeshId mesh, Vec3 translation, Quat rotation, Vec3 scale) {
	Vec3 lower, upper;
	NvFlexGetTriangleMeshBounds(g_flexLib, mesh, lower, upper);

	NvFlexCollisionGeometry geo;
	geo.triMesh.mesh = mesh;
	geo.triMesh.scale[0] = scale.x;
	geo.triMesh.scale[1] = scale.y;
	geo.triMesh.scale[2] = scale.z;

	g_buffers->shapePositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapeRotations.push_back(Quat(rotation));
	g_buffers->shapePrevPositions.push_back(Vec4(translation, 0.0f));
	g_buffers->shapePrevRotations.push_back(Quat(rotation));
	g_buffers->shapeGeometry.push_back((NvFlexCollisionGeometry&)geo);
	g_buffers->shapeFlags.push_back(NvFlexMakeShapeFlags(eNvFlexShapeTriangleMesh, false));
}

void GetParticleBounds(SimBuffers *g_buffers, Vec3& lower, Vec3& upper) {
	lower = Vec3(FLT_MAX);
	upper = Vec3(-FLT_MAX);

	for (int i=0; i < g_buffers->positions.size(); ++i) {
		lower = Min(Vec3(g_buffers->positions[i]), lower);
		upper = Max(Vec3(g_buffers->positions[i]), upper);
	}
}

// calculates the union bounds of all the collision shapes in the scene
void GetShapeBounds(NvFlexLibrary* g_flexLib, SimBuffers *g_buffers, Vec3& totalLower, Vec3& totalUpper) {
	Bounds totalBounds;

	for (int i=0; i < g_buffers->shapeFlags.size(); ++i) {
		NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

		int type = g_buffers->shapeFlags[i]&eNvFlexShapeFlagTypeMask;

		Vec3 localLower;
		Vec3 localUpper;

		switch(type) {
			case eNvFlexShapeBox: {
				localLower = -Vec3(geo.box.halfExtents);
				localUpper = Vec3(geo.box.halfExtents);
				break;
			}
			case eNvFlexShapeSphere: {
				localLower = -geo.sphere.radius;
				localUpper = geo.sphere.radius;
				break;
			}
			case eNvFlexShapeCapsule: {
				localLower = -Vec3(geo.capsule.halfHeight, 0.0f, 0.0f) - Vec3(geo.capsule.radius);
				localUpper = Vec3(geo.capsule.halfHeight, 0.0f, 0.0f) + Vec3(geo.capsule.radius);
				break;
			}
			case eNvFlexShapeConvexMesh: {
				NvFlexGetConvexMeshBounds(g_flexLib, geo.convexMesh.mesh, localLower, localUpper);

				// apply instance scaling
				localLower *= geo.convexMesh.scale;
				localUpper *= geo.convexMesh.scale;
				break;
			}
			case eNvFlexShapeTriangleMesh: {
				NvFlexGetTriangleMeshBounds(g_flexLib, geo.triMesh.mesh, localLower, localUpper);
				
				// apply instance scaling
				localLower *= Vec3(geo.triMesh.scale);
				localUpper *= Vec3(geo.triMesh.scale);
				break;
			}
			case eNvFlexShapeSDF: {
				localLower = 0.0f;
				localUpper = geo.sdf.scale;
				break;
			}
		};

		// transform local bounds to world space
		Vec3 worldLower, worldUpper;
		TransformBounds(localLower, localUpper, Vec3(g_buffers->shapePositions[i]), g_buffers->shapeRotations[i], 1.0f, worldLower, worldUpper);

		totalBounds = Union(totalBounds, Bounds(worldLower, worldUpper));
	}

	totalLower = totalBounds.lower;
	totalUpper = totalBounds.upper;
}

// calculates the center of mass of every rigid given a set of particle positions and rigid indices
void CalculateRigidCentersOfMass(const Vec4* restPositions, int numRestPositions, const int* offsets, Vec3* translations, const int* indices, int numRigids) {
	// To improve the accuracy of the result, first transform the restPositions to relative coordinates (by finding the mean and subtracting that from all positions)
	// Note: If this is not done, one might see ghost forces if the mean of the restPositions is far from the origin.
	Vec3 shapeOffset(0.0f);

	for (int i = 0; i < numRestPositions; i++) {
		shapeOffset += Vec3(restPositions[i]);
	}

	shapeOffset /= float(numRestPositions);

	for (int i=0; i < numRigids; ++i) {
		const int startIndex = offsets[i];
		const int endIndex = offsets[i+1];

		const int n = endIndex-startIndex;

		assert(n);

		Vec3 com;
	
		for (int j=startIndex; j < endIndex; ++j) {
			const int r = indices[j];

			// By subtracting shapeOffset the calculation is done in relative coordinates
			com += Vec3(restPositions[r]) - shapeOffset;
		}

		com /= float(n);

		// Add the shapeOffset to switch back to absolute coordinates
		com += shapeOffset;

		translations[i] = com;

	}
}

// calculates local space positions given a set of particle positions, rigid indices and centers of mass of the rigids
void CalculateRigidLocalPositions(const Vec4* restPositions, const int* offsets, const Vec3* translations, const int* indices, int numRigids, Vec3* localPositions) {
	int count = 0;

	for (int i=0; i < numRigids; ++i) {
		const int startIndex = offsets[i];
		const int endIndex = offsets[i+1];

		assert(endIndex-startIndex);

		for (int j=startIndex; j < endIndex; ++j) {
			const int r = indices[j];

			localPositions[count++] = Vec3(restPositions[r]) - translations[i];
		}
	}
}

inline float sqr(float x) { return x*x; }

