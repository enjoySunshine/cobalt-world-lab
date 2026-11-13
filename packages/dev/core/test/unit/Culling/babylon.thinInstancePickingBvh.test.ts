import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { NullEngine } from "core/Engines/nullEngine";
import { Matrix, Vector3 } from "core/Maths/math.vector";
import { Mesh } from "core/Meshes/mesh";
import { MeshBuilder } from "core/Meshes/meshBuilder";
import { Scene } from "core/scene";
import { Ray, PickingCustomization } from "core/Culling/ray";
import { PickingInfo } from "core/Collisions/pickingInfo";
import "core/Materials/standardMaterial";
import "core/Meshes/thinInstanceMesh";

describe("ThinInstance picking BVH", () => {
    let engine: NullEngine;
    let scene: Scene;
    let box: Mesh;

    beforeEach(() => {
        engine = new NullEngine({
            renderHeight: 256,
            renderWidth: 256,
            textureSize: 256,
            deterministicLockstep: false,
            lockstepMaxSteps: 1,
        });
        // NullEngine reports no instancedArrays support by default; thinInstanceAdd requires it.
        engine.getCaps().instancedArrays = true;
        scene = new Scene(engine);
        box = MeshBuilder.CreateBox("box", { size: 1 }, scene);
    });

    afterEach(() => {
        scene.dispose();
        engine.dispose();
    });

    /**
     * Builds the warehouse layout from the performance brief: 100 columns x 50
     * rows (5000 crates), spacing 2 on both axes, 1-unit boxes hanging on one
     * grid. The whole batch is shifted by the given parent world x.
     */
    function buildWarehouse(parentWorldX: number): void {
        const columns = 100;
        const rows = 50;
        const matrices = new Float32Array(columns * rows * 16);
        const matrix = new Matrix();

        let index = 0;
        for (let row = 0; row < rows; row++) {
            for (let column = 0; column < columns; column++) {
                Matrix.TranslationToRef(column * 2, 0, row * 2, matrix);
                matrix.copyToArray(matrices, index * 16);
                index++;
            }
        }

        box.thinInstanceSetBuffer("matrix", matrices, 16, false);
        box.thinInstanceEnablePicking = true;
        box.position.x = parentWorldX;
        box.computeWorldMatrix(true);
    }

    it("picks crate 505 in a 5000-instance warehouse with the expected distance", () => {
        buildWarehouse(100);

        // Instance 505 sits at local (10, 10); world (110, 10). Ray from y=8 down.
        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        const info = scene.pickWithRay(ray);

        expect(info!.hit).toBe(true);
        expect(info!.thinInstanceIndex).toBe(505);
        expect(info!.distance).toBeCloseTo(7.5, 5);
    });

    it("picks the nearer of two overlapping crates", () => {
        // 21 instances: index 10 at local (20,0,0) and index 20 at local (20,3,0),
        // matching the warehouse brief; remaining instances stay out of the ray path.
        const matrices = new Float32Array(21 * 16);
        Matrix.Translation(20, 0, 0).copyToArray(matrices, 10 * 16);
        Matrix.Translation(20, 3, 0).copyToArray(matrices, 20 * 16);

        box.thinInstanceSetBuffer("matrix", matrices, 16, false);
        box.thinInstanceEnablePicking = true;
        box.position.x = 100;
        box.computeWorldMatrix(true);

        const ray = new Ray(new Vector3(120, 8, 0), new Vector3(0, -1, 0), 30);
        const info = scene.pickWithRay(ray);

        expect(info!.hit).toBe(true);
        expect(info!.thinInstanceIndex).toBe(20);
        expect(info!.distance).toBeCloseTo(4.5, 5);
    });

    it("picks instance 0 at the parent origin", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(100, 8, 0), new Vector3(0, -1, 0), 30);
        const info = scene.pickWithRay(ray);

        expect(info!.hit).toBe(true);
        expect(info!.thinInstanceIndex).toBe(0);
        expect(info!.distance).toBeCloseTo(7.5, 5);
    });

    it("reports a clean miss on an empty ray", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(90, 8, -5), new Vector3(0, -1, 0), 30);
        const info = scene.pickWithRay(ray);

        expect(info!.hit).toBe(false);
        expect(info!.thinInstanceIndex).toBe(-1);
    });

    it("keeps fastCheck results identical to the linear scan", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        const fastInfo = scene.pickWithRay(ray, undefined, true);
        const closestInfo = scene.pickWithRay(ray, undefined, false);

        expect(fastInfo!.hit).toBe(true);
        expect(fastInfo!.thinInstanceIndex).toBe(closestInfo!.thinInstanceIndex);
        expect(fastInfo!.distance).toBeCloseTo(closestInfo!.distance, 5);
    });

    it("multi-picks every crate pierced by a horizontal ray", () => {
        buildWarehouse(100);

        // Horizontal ray along x at local y=0, z=0: it crosses all 100 columns of the first row.
        const ray = new Ray(new Vector3(99, 0, 0), new Vector3(1, 0, 0));
        const infos = scene.multiPickWithRay(ray);

        expect(infos).toBeTruthy();
        expect(infos!.length).toBe(100);
        const indices = infos!.map((i) => i.thinInstanceIndex).sort((a, b) => a - b);
        for (let column = 0; column < 100; column++) {
            expect(indices[column]).toBe(column);
        }
    });

    it("rebuilds the BVH after in-place matrix mutations", () => {
        buildWarehouse(100);

        // First pick builds and caches the BVH.
        const firstRay = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        expect(scene.pickWithRay(firstRay)!.thinInstanceIndex).toBe(505);

        // Move instance 505 from local (10,10) to (30,10) and refresh the buffer.
        const matrixData = box._thinInstanceDataStorage.matrixData!;
        matrixData[505 * 16 + 12] = 30;
        box.thinInstanceBufferUpdated("matrix");

        const oldPositionRay = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        expect(scene.pickWithRay(oldPositionRay)!.hit).toBe(false);

        const newPositionRay = new Ray(new Vector3(130, 8, 10), new Vector3(0, -1, 0), 30);
        const movedInfo = scene.pickWithRay(newPositionRay);
        expect(movedInfo!.hit).toBe(true);
        expect(movedInfo!.thinInstanceIndex).toBe(505);
    });

    it("rebuilds the BVH after thinInstanceSetMatrixAt", () => {
        buildWarehouse(100);

        expect(scene.pickWithRay(new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30))!.thinInstanceIndex).toBe(505);

        box.thinInstanceSetMatrixAt(505, Matrix.Translation(50, 0, 10), true);

        const oldPositionRay = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        expect(scene.pickWithRay(oldPositionRay)!.hit).toBe(false);

        const newPositionRay = new Ray(new Vector3(150, 8, 10), new Vector3(0, -1, 0), 30);
        expect(scene.pickWithRay(newPositionRay)!.thinInstanceIndex).toBe(505);
    });

    it("rebuilds the BVH when the visible instance count is reduced", () => {
        buildWarehouse(100);

        expect(scene.pickWithRay(new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30))!.thinInstanceIndex).toBe(505);

        // Hide everything above index 100: instance 505 must no longer be pickable.
        box.thinInstanceCount = 100;

        const info = scene.pickWithRay(new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30));
        expect(info!.hit).toBe(false);

        // Instance 0 still picks.
        expect(scene.pickWithRay(new Ray(new Vector3(100, 8, 0), new Vector3(0, -1, 0), 30))!.thinInstanceIndex).toBe(0);
    });

    it("honors a predicate that filters BVH candidates", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        // The mesh-level predicate call uses thinInstanceIndex -1 and must be accepted.
        const blocked = scene.pickWithRay(ray, (_mesh, thinInstanceIndex) => thinInstanceIndex === -1 || thinInstanceIndex !== 505);
        expect(blocked!.hit).toBe(false);

        const allowed = scene.pickWithRay(ray, (_mesh, thinInstanceIndex) => thinInstanceIndex === -1 || thinInstanceIndex === 505);
        expect(allowed!.hit).toBe(true);
        expect(allowed!.thinInstanceIndex).toBe(505);
    });

    it("caches the BVH across picks and invalidates it on matrix changes", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        scene.pickWithRay(ray);

        const firstBVH = box._thinInstanceDataStorage.pickingBVH;
        expect(firstBVH).not.toBeNull();
        expect(firstBVH!.instanceCount).toBe(5000);

        scene.pickWithRay(ray);
        expect(box._thinInstanceDataStorage.pickingBVH).toBe(firstBVH);

        box.thinInstanceSetMatrixAt(1, Matrix.Translation(2, 0, 0), true);
        expect(box._thinInstanceDataStorage.pickingBVH).toBeNull();

        scene.pickWithRay(ray);
        expect(box._thinInstanceDataStorage.pickingBVH).not.toBeNull();
        expect(box._thinInstanceDataStorage.pickingBVH).not.toBe(firstBVH);
    });

    it("picks rotated thin instances through the BVH", () => {
        const matrices = new Float32Array(3 * 16);
        Matrix.Identity().copyToArray(matrices, 0);
        Matrix.RotationY(Math.PI / 4)
            .multiply(Matrix.Translation(4, 0, 0))
            .copyToArray(matrices, 16);
        Matrix.Translation(8, 0, 0).copyToArray(matrices, 32);

        box.thinInstanceSetBuffer("matrix", matrices, 16, false);
        box.thinInstanceEnablePicking = true;
        box.computeWorldMatrix(true);

        const ray = new Ray(new Vector3(4, 5, 0), new Vector3(0, -1, 0));
        const info = scene.pickWithRay(ray);

        expect(info!.hit).toBe(true);
        expect(info!.thinInstanceIndex).toBe(1);
    });

    it("completes a 5000-instance pick well below the 1.2ms budget", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);

        // Warm up: builds the BVH and lets the JIT settle.
        for (let i = 0; i < 5; i++) {
            scene.pickWithRay(ray);
        }

        const iterations = 50;
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            const info = scene.pickWithRay(ray);
            expect(info!.thinInstanceIndex).toBe(505);
        }
        const averageMs = (performance.now() - start) / iterations;

        // The brief requires 1.2ms average; leave headroom for CI variance.
        expect(averageMs).toBeLessThan(1.2);
    });

    it("keeps empty-ray picks effectively free", () => {
        buildWarehouse(100);

        const ray = new Ray(new Vector3(90, 8, -5), new Vector3(0, -1, 0), 30);

        for (let i = 0; i < 5; i++) {
            scene.pickWithRay(ray);
        }

        const iterations = 50;
        const start = performance.now();
        for (let i = 0; i < iterations; i++) {
            const info = scene.pickWithRay(ray);
            expect(info!.hit).toBe(false);
            expect(info!.thinInstanceIndex).toBe(-1);
        }
        const averageMs = (performance.now() - start) / iterations;

        // An empty ray is rejected by the outer mesh bounds before any BVH work;
        // it must stay an order of magnitude cheaper than a real pick (budget 1.2ms).
        expect(averageMs).toBeLessThan(0.5);
    });

    it("picks a single-instance mesh through the BVH", () => {
        box.thinInstanceAdd(Matrix.Translation(3, 0, 2), true);
        box.thinInstanceEnablePicking = true;
        box.computeWorldMatrix(true);

        const ray = new Ray(new Vector3(3, 5, 2), new Vector3(0, -1, 0));
        const info = scene.pickWithRay(ray);

        expect(info!.hit).toBe(true);
        expect(info!.thinInstanceIndex).toBe(0);
    });

    it("falls back to the linear scan when a custom picker is installed", () => {
        buildWarehouse(100);

        let instancePickerCalls = 0;
        PickingCustomization.internalPickerForMesh = (_pickingInfo, _rayFunction, mesh, _world, _fastCheck, onlyBoundingInfo) => {
            if (onlyBoundingInfo) {
                const result = new PickingInfo();
                result.hit = true;
                result.pickedMesh = mesh;
                return result;
            }
            instancePickerCalls++;
            return null!;
        };

        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        const info = scene.pickWithRay(ray);

        // No BVH is built while a custom picker is active...
        expect(box._thinInstanceDataStorage.pickingBVH).toBeNull();
        // ...and every instance is offered to the custom picker (linear scan).
        expect(instancePickerCalls).toBe(5000);
        expect(info!.hit).toBe(false);

        PickingCustomization.internalPickerForMesh = undefined;
    });

    it("agrees with the raw-bounding-info-less linear scan on a grid of rays", () => {
        buildWarehouse(100);

        const rays: Ray[] = [];
        for (let column = 0; column < 100; column += 11) {
            for (let row = 0; row < 50; row += 7) {
                rays.push(new Ray(new Vector3(100 + column * 2 + 0.3, 8, row * 2 - 0.3), new Vector3(0, -1, 0), 30));
            }
        }

        // BVH path results.
        const bvhResults = rays.map((ray) => {
            const info = scene.pickWithRay(ray);
            return { hit: info!.hit, index: info!.thinInstanceIndex, distance: info!.distance };
        });

        // Force the linear path by removing rawBoundingInfo (which disables the BVH).
        box.rawBoundingInfo = null;
        box._thinInstanceDataStorage.pickingBVH = null;
        const linearResults = rays.map((ray) => {
            const info = scene.pickWithRay(ray);
            return { hit: info!.hit, index: info!.thinInstanceIndex, distance: info!.distance };
        });

        expect(bvhResults.length).toBe(linearResults.length);
        for (let i = 0; i < rays.length; i++) {
            expect(bvhResults[i].hit).toBe(linearResults[i].hit);
            expect(bvhResults[i].index).toBe(linearResults[i].index);
            if (bvhResults[i].hit) {
                expect(bvhResults[i].distance).toBeCloseTo(linearResults[i].distance, 6);
            }
        }
    });

    it("rebuilds after replacing the whole matrix buffer", () => {
        buildWarehouse(100);
        const ray = new Ray(new Vector3(110, 8, 10), new Vector3(0, -1, 0), 30);
        expect(scene.pickWithRay(ray)!.thinInstanceIndex).toBe(505);

        // Replace with a single instance at a different location.
        const replacement = new Float32Array(16);
        Matrix.Translation(40, 0, 40).copyToArray(replacement, 0);
        box.thinInstanceSetBuffer("matrix", replacement, 16, false);

        expect(box._thinInstanceDataStorage.pickingBVH).toBeNull();
        const movedRay = new Ray(new Vector3(140, 8, 40), new Vector3(0, -1, 0), 30);
        const info = scene.pickWithRay(movedRay);
        expect(info!.hit).toBe(true);
        expect(info!.thinInstanceIndex).toBe(0);
        expect(box._thinInstanceDataStorage.pickingBVH!.instanceCount).toBe(1);
    });
});
