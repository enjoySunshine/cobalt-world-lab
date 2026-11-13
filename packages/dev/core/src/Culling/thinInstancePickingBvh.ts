/** This file must only contain pure code and pure imports */

import { Matrix, TmpVectors, Vector3 } from "core/Maths/math.vector.pure";
import type { Ray } from "./ray.core";

const LeafSize = 4;
// Extent-relative expansion applied to every node box so the BVH precheck can
// never reject an instance that the per-instance raw box/sphere precheck accepts.
const BoundsEpsilon = 1e-4;
const AbsoluteBoundsEpsilon = 1e-6;
const ParallelRayEpsilon = 0.0000001;

/** Minimal structural type for a read-only 3D vector so the builder stays decoupled. */
interface Vector3Like {
    x: number;
    y: number;
    z: number;
}

/**
 * Per-mesh cache of a BVH built over the local-space axis-aligned bounding
 * boxes of a mesh's thin instances. The BVH lives in the mesh's own local
 * space so that moving the mesh (changing its world matrix) does not require
 * a rebuild: only the picking ray is transformed into local space once per
 * pick.
 * @internal
 */
export class ThinInstancePickingBVH {
    /** Flat node bounds. Node `i` occupies offsets `i * 6 .. i * 6 + 5` (minX, minY, minZ, maxX, maxY, maxZ). */
    public readonly bounds: Float32Array;
    /** Internal nodes only: left child node index. Leaf nodes hold `-(leafStart + 1)` (always negative). */
    public readonly left: Int32Array;
    /** Internal nodes only: right child node index. Unused for leaves. */
    public readonly right: Int32Array;
    /** Contiguous instance indices; leaves reference a range of this array. */
    public readonly indices: Uint32Array;
    /** Leaf nodes only: start offset (inclusive) into {@link indices}. */
    public readonly leafStart: Int32Array;
    /** Leaf nodes only: number of instances covered by the leaf. */
    public readonly leafCount: Int32Array;
    /** Number of nodes (internal + leaves) in {@link bounds} / {@link left}. */
    public readonly nodeCount: number;
    /** Number of thin instances the BVH was built from. */
    public readonly instanceCount: number;
    /** Matrix-data version this BVH was built from; used to detect stale caches. */
    public matrixDataVersion: number;
    /** Per-instance transformed AABBs in mesh-local space, 6 floats per instance (minX, minY, minZ, maxX, maxY, maxZ). */
    public readonly instanceBounds: Float32Array;

    /**
     * Builds a BVH from the thin-instance matrices in `matrixData`.
     * Each instance box is the mesh's raw bounding box transformed by the
     * instance matrix (all in mesh-local space).
     * @param matrixData column-major 4x4 instance matrices, 16 floats per instance
     * @param instanceCount number of valid instances in `matrixData`
     * @param rawMin mesh raw bounding-box minimum (mesh-local space)
     * @param rawMax mesh raw bounding-box maximum (mesh-local space)
     * @param matrixDataVersion matrix-data version to stamp on the cache
     * @returns the built BVH
     */
    public static Build(matrixData: Float32Array, instanceCount: number, rawMin: Vector3Like, rawMax: Vector3Like, matrixDataVersion: number): ThinInstancePickingBVH {
        // Per-instance transformed AABBs, 6 floats each.
        const instanceBounds = new Float32Array(instanceCount * 6);
        const matrix = TmpVectors.Matrix[0];
        const corner = TmpVectors.Vector3[7];
        const transformed = TmpVectors.Vector3[8];

        for (let index = 0; index < instanceCount; index++) {
            Matrix.FromArrayToRef(matrixData, index * 16, matrix);

            const offset = index * 6;
            let minX = Number.POSITIVE_INFINITY;
            let minY = Number.POSITIVE_INFINITY;
            let minZ = Number.POSITIVE_INFINITY;
            let maxX = Number.NEGATIVE_INFINITY;
            let maxY = Number.NEGATIVE_INFINITY;
            let maxZ = Number.NEGATIVE_INFINITY;

            for (let cx = 0; cx < 2; cx++) {
                for (let cy = 0; cy < 2; cy++) {
                    for (let cz = 0; cz < 2; cz++) {
                        corner.set(cx ? rawMax.x : rawMin.x, cy ? rawMax.y : rawMin.y, cz ? rawMax.z : rawMin.z);
                        Vector3.TransformCoordinatesToRef(corner, matrix, transformed);

                        if (transformed.x < minX) {
                            minX = transformed.x;
                        }
                        if (transformed.y < minY) {
                            minY = transformed.y;
                        }
                        if (transformed.z < minZ) {
                            minZ = transformed.z;
                        }
                        if (transformed.x > maxX) {
                            maxX = transformed.x;
                        }
                        if (transformed.y > maxY) {
                            maxY = transformed.y;
                        }
                        if (transformed.z > maxZ) {
                            maxZ = transformed.z;
                        }
                    }
                }
            }

            instanceBounds[offset] = minX;
            instanceBounds[offset + 1] = minY;
            instanceBounds[offset + 2] = minZ;
            instanceBounds[offset + 3] = maxX;
            instanceBounds[offset + 4] = maxY;
            instanceBounds[offset + 5] = maxZ;
        }

        const indices = new Uint32Array(instanceCount);
        for (let index = 0; index < instanceCount; index++) {
            indices[index] = index;
        }

        // Upper bound on nodes for an arbitrary binary tree over N instances:
        // N-1 internal nodes + N singleton leaves = 2N-1. Leaves usually hold
        // several instances, so the real count is far lower, but the arrays are
        // sized for the worst case because TypedArrays silently drop out-of-range
        // writes (an unbalanced median split would otherwise produce empty slots).
        const maxNodes = instanceCount > 0 ? 2 * instanceCount : 0;
        const bounds = new Float32Array(maxNodes * 6);
        const left = new Int32Array(maxNodes);
        const right = new Int32Array(maxNodes);
        const leafStart = new Int32Array(maxNodes);
        const leafCount = new Int32Array(maxNodes);

        const builder = new _ThinInstanceBVHBuilder(instanceBounds, indices, bounds, left, right, leafStart, leafCount);
        const nodeCount = builder.build();

        // Expand every node box by a tiny extent-relative margin.
        for (let node = 0; node < nodeCount; node++) {
            const offset = node * 6;
            const ex = Math.max(AbsoluteBoundsEpsilon, (bounds[offset + 3] - bounds[offset]) * BoundsEpsilon);
            const ey = Math.max(AbsoluteBoundsEpsilon, (bounds[offset + 4] - bounds[offset + 1]) * BoundsEpsilon);
            const ez = Math.max(AbsoluteBoundsEpsilon, (bounds[offset + 5] - bounds[offset + 2]) * BoundsEpsilon);
            bounds[offset] -= ex;
            bounds[offset + 1] -= ey;
            bounds[offset + 2] -= ez;
            bounds[offset + 3] += ex;
            bounds[offset + 4] += ey;
            bounds[offset + 5] += ez;
        }

        return new ThinInstancePickingBVH(bounds, left, right, indices, leafStart, leafCount, nodeCount, instanceCount, matrixDataVersion, instanceBounds);
    }

    private constructor(
        bounds: Float32Array,
        left: Int32Array,
        right: Int32Array,
        indices: Uint32Array,
        leafStart: Int32Array,
        leafCount: Int32Array,
        nodeCount: number,
        instanceCount: number,
        matrixDataVersion: number,
        instanceBounds: Float32Array
    ) {
        this.bounds = bounds;
        this.left = left;
        this.right = right;
        this.indices = indices;
        this.leafStart = leafStart;
        this.leafCount = leafCount;
        this.nodeCount = nodeCount;
        this.instanceCount = instanceCount;
        this.matrixDataVersion = matrixDataVersion;
        this.instanceBounds = instanceBounds;
    }

    /**
     * Tests the mesh-local ray against a single instance's local-space AABB.
     * This is a cheap precheck used to skip the full per-instance matrix
     * transform, ray inversion and triangle test when the ray clearly misses.
     * Ray length is ignored, matching the bounds precheck semantics.
     * @param ray mesh-local picking ray
     * @param instanceIndex index of the instance whose box is tested
     * @param intersectionThreshold extra extent added to the box on every side (used by line meshes)
     */
    public rayIntersectsInstanceBox(ray: Ray, instanceIndex: number, intersectionThreshold: number = 0): boolean {
        return (
            rayBoxEntryDistance(
                this.instanceBounds,
                instanceIndex * 6,
                ray.origin.x,
                ray.origin.y,
                ray.origin.z,
                ray.direction.x,
                ray.direction.y,
                ray.direction.z,
                intersectionThreshold
            ) !== Infinity
        );
    }

    /**
     * Collects the indices of every instance whose leaf box may be hit by `ray`.
     * The indices are returned in ascending order so iteration order matches
     * the linear scan the BVH replaces (in particular fastCheck keeps picking
     * the lowest-indexed hit and multi-pick stays index ordered).
     * @param ray ray expressed in the mesh-local space the BVH was built in
     * @param target array receiving the instance indices (cleared first)
     * @returns the number of indices written into `target`
     */
    public query(ray: Ray, target: number[]): number {
        target.length = 0;

        if (this.nodeCount === 0) {
            return 0;
        }

        const stackNodes = ThinInstancePickingBVH._StackNodes;
        let stackSize = 0;
        stackNodes[stackSize++] = 0;

        const ox = ray.origin.x;
        const oy = ray.origin.y;
        const oz = ray.origin.z;
        const dx = ray.direction.x;
        const dy = ray.direction.y;
        const dz = ray.direction.z;

        let count = 0;

        while (stackSize > 0) {
            const node = stackNodes[--stackSize];

            if (this.left[node] < 0) {
                const start = this.leafStart[node];
                const end = start + this.leafCount[node];
                for (let i = start; i < end; i++) {
                    target[count++] = this.indices[i];
                }
                continue;
            }

            const leftNode = this.left[node];
            const rightNode = this.right[node];

            const leftDistance = rayBoxEntryDistance(this.bounds, leftNode * 6, ox, oy, oz, dx, dy, dz);
            const rightDistance = rayBoxEntryDistance(this.bounds, rightNode * 6, ox, oy, oz, dx, dy, dz);

            if (leftDistance === Infinity) {
                if (rightDistance !== Infinity) {
                    stackNodes[stackSize++] = rightNode;
                }
            } else if (rightDistance === Infinity) {
                stackNodes[stackSize++] = leftNode;
            } else {
                // Both children are candidates; push order does not affect the result
                // because candidates are sorted by instance index before returning.
                stackNodes[stackSize++] = rightNode;
                stackNodes[stackSize++] = leftNode;
            }
        }

        if (count > 1) {
            target.length = count;
            target.sort((a, b) => a - b);
        }

        return count;
    }

    private static _StackNodes: number[] = [];
}

/**
 * Slab test returning the ray entry distance (tMin) or +Infinity on a miss.
 * Mirrors {@link Ray.intersectsBoxMinMax} but also reports the entry distance.
 * Ray length is intentionally ignored, matching the existing bounds precheck.
 * @param intersectionThreshold extra extent added to each box face (0 for node traversal)
 */
function rayBoxEntryDistance(
    bounds: Float32Array,
    offset: number,
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    intersectionThreshold: number = 0
): number {
    let tMin = Number.NEGATIVE_INFINITY;
    let tMax = Number.POSITIVE_INFINITY;

    // X slab
    if (Math.abs(dx) < ParallelRayEpsilon) {
        if (ox < bounds[offset] - ParallelRayEpsilon - intersectionThreshold || ox > bounds[offset + 3] + ParallelRayEpsilon + intersectionThreshold) {
            return Infinity;
        }
    } else {
        const inv = 1.0 / dx;
        let near = (bounds[offset] - intersectionThreshold - ox) * inv;
        let far = (bounds[offset + 3] + intersectionThreshold - ox) * inv;
        if (near > far) {
            const tmp = near;
            near = far;
            far = tmp;
        }
        if (near > tMin) {
            tMin = near;
        }
        if (far < tMax) {
            tMax = far;
        }
        if (tMin > tMax) {
            return Infinity;
        }
    }

    // Y slab
    if (Math.abs(dy) < ParallelRayEpsilon) {
        if (oy < bounds[offset + 1] - ParallelRayEpsilon - intersectionThreshold || oy > bounds[offset + 4] + ParallelRayEpsilon + intersectionThreshold) {
            return Infinity;
        }
    } else {
        const inv = 1.0 / dy;
        let near = (bounds[offset + 1] - intersectionThreshold - oy) * inv;
        let far = (bounds[offset + 4] + intersectionThreshold - oy) * inv;
        if (near > far) {
            const tmp = near;
            near = far;
            far = tmp;
        }
        if (near > tMin) {
            tMin = near;
        }
        if (far < tMax) {
            tMax = far;
        }
        if (tMin > tMax) {
            return Infinity;
        }
    }

    // Z slab
    if (Math.abs(dz) < ParallelRayEpsilon) {
        if (oz < bounds[offset + 2] - ParallelRayEpsilon - intersectionThreshold || oz > bounds[offset + 5] + ParallelRayEpsilon + intersectionThreshold) {
            return Infinity;
        }
    } else {
        const inv = 1.0 / dz;
        let near = (bounds[offset + 2] - intersectionThreshold - oz) * inv;
        let far = (bounds[offset + 5] + intersectionThreshold - oz) * inv;
        if (near > far) {
            const tmp = near;
            near = far;
            far = tmp;
        }
        if (near > tMin) {
            tMin = near;
        }
        if (far < tMax) {
            tMax = far;
        }
        if (tMin > tMax) {
            return Infinity;
        }
    }

    return tMin;
}

/**
 * Recursive median-split BVH builder over per-instance AABBs.
 */
class _ThinInstanceBVHBuilder {
    private readonly _instanceBounds: Float32Array;
    private readonly _indices: Uint32Array;
    private readonly _bounds: Float32Array;
    private readonly _left: Int32Array;
    private readonly _right: Int32Array;
    private readonly _leafStart: Int32Array;
    private readonly _leafCount: Int32Array;
    private _nodeCount = 0;

    constructor(instanceBounds: Float32Array, indices: Uint32Array, bounds: Float32Array, left: Int32Array, right: Int32Array, leafStart: Int32Array, leafCount: Int32Array) {
        this._instanceBounds = instanceBounds;
        this._indices = indices;
        this._bounds = bounds;
        this._left = left;
        this._right = right;
        this._leafStart = leafStart;
        this._leafCount = leafCount;
    }

    public build(): number {
        this._buildRecursive(0, this._indices.length);
        return this._nodeCount;
    }

    private _buildRecursive(start: number, end: number): number {
        const node = this._nodeCount++;
        const count = end - start;

        this._computeNodeBounds(node, start, end);

        if (count <= LeafSize) {
            // Leaf: negative left value identifies the leaf; the instance range is stored separately.
            this._left[node] = -(start + 1);
            this._leafStart[node] = start;
            this._leafCount[node] = count;
            return node;
        }

        const split = this._medianSplit(start, end);

        // Degenerate ranges (all centroids equal on every axis, or a median
        // landing on a boundary) cannot make progress: keep the whole range as
        // one leaf instead of recursing into an empty half.
        if (split <= start || split >= end) {
            this._left[node] = -(start + 1);
            this._leafStart[node] = start;
            this._leafCount[node] = count;
            return node;
        }

        const leftNode = this._buildRecursive(start, split);
        const rightNode = this._buildRecursive(split, end);

        this._left[node] = leftNode;
        this._right[node] = rightNode;

        return node;
    }

    private _computeNodeBounds(node: number, start: number, end: number): void {
        const offset = node * 6;
        let minX = Number.POSITIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let minZ = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        let maxZ = Number.NEGATIVE_INFINITY;

        for (let i = start; i < end; i++) {
            const b = this._indices[i] * 6;
            if (this._instanceBounds[b] < minX) {
                minX = this._instanceBounds[b];
            }
            if (this._instanceBounds[b + 1] < minY) {
                minY = this._instanceBounds[b + 1];
            }
            if (this._instanceBounds[b + 2] < minZ) {
                minZ = this._instanceBounds[b + 2];
            }
            if (this._instanceBounds[b + 3] > maxX) {
                maxX = this._instanceBounds[b + 3];
            }
            if (this._instanceBounds[b + 4] > maxY) {
                maxY = this._instanceBounds[b + 4];
            }
            if (this._instanceBounds[b + 5] > maxZ) {
                maxZ = this._instanceBounds[b + 5];
            }
        }

        this._bounds[offset] = minX;
        this._bounds[offset + 1] = minY;
        this._bounds[offset + 2] = minZ;
        this._bounds[offset + 3] = maxX;
        this._bounds[offset + 4] = maxY;
        this._bounds[offset + 5] = maxZ;
    }

    /**
     * Partitions the index range around the median centroid on the
     * largest-extent axis and returns the median split position. Everything
     * before the split has a centroid <= the median, everything after >= it.
     * Uses median-of-three quickselect with a three-way partition so ranges
     * full of equal centroids stay linear instead of quadratic.
     */
    private _medianSplit(start: number, end: number): number {
        let centroidMinX = Infinity;
        let centroidMinY = Infinity;
        let centroidMinZ = Infinity;
        let centroidMaxX = -Infinity;
        let centroidMaxY = -Infinity;
        let centroidMaxZ = -Infinity;

        for (let i = start; i < end; i++) {
            const b = this._indices[i] * 6;
            const cx = (this._instanceBounds[b] + this._instanceBounds[b + 3]) * 0.5;
            const cy = (this._instanceBounds[b + 1] + this._instanceBounds[b + 4]) * 0.5;
            const cz = (this._instanceBounds[b + 2] + this._instanceBounds[b + 5]) * 0.5;
            if (cx < centroidMinX) {
                centroidMinX = cx;
            }
            if (cy < centroidMinY) {
                centroidMinY = cy;
            }
            if (cz < centroidMinZ) {
                centroidMinZ = cz;
            }
            if (cx > centroidMaxX) {
                centroidMaxX = cx;
            }
            if (cy > centroidMaxY) {
                centroidMaxY = cy;
            }
            if (cz > centroidMaxZ) {
                centroidMaxZ = cz;
            }
        }

        const extentX = centroidMaxX - centroidMinX;
        const extentY = centroidMaxY - centroidMinY;
        const extentZ = centroidMaxZ - centroidMinZ;
        const axis = extentX >= extentY && extentX >= extentZ ? 0 : extentY >= extentZ ? 1 : 2;
        const centroidOffset = axis * 2;
        const medianTarget = start + Math.floor((end - start) / 2);

        this._quickSelect(start, end - 1, medianTarget, centroidOffset);

        return medianTarget;
    }

    /** Centroid value of instance `index` on the chosen axis component. */
    private _centroid(index: number, centroidOffset: number): number {
        const b = index * 6;
        return (this._instanceBounds[b + centroidOffset] + this._instanceBounds[b + centroidOffset + 3]) * 0.5;
    }

    /**
     * Rearranges indices in [left, right] so the element at `target` is the
     * one that would be there in sorted order, with smaller values before and
     * larger values after. Three-way Dutch-flag partition groups equals.
     */
    private _quickSelect(left: number, right: number, target: number, centroidOffset: number): void {
        const indices = this._indices;

        while (left < right) {
            const pivotValue = this._choosePivotValue(left, right, centroidOffset);

            // Three-way partition: [left .. lt-1] < pivot, [lt .. gt] == pivot, [gt+1 .. right] > pivot
            let lt = left;
            let i = left;
            let gt = right;
            while (i <= gt) {
                const value = this._centroid(indices[i], centroidOffset);
                if (value < pivotValue) {
                    this._swap(lt, i);
                    lt++;
                    i++;
                } else if (value > pivotValue) {
                    this._swap(i, gt);
                    gt--;
                } else {
                    i++;
                }
            }

            if (target < lt) {
                right = lt - 1;
            } else if (target > gt) {
                left = gt + 1;
            } else {
                return;
            }
        }
    }

    /** Median-of-three pivot centroid value, avoiding worst-case behavior on sorted ranges. */
    private _choosePivotValue(left: number, right: number, centroidOffset: number): number {
        const middle = (left + right) >> 1;
        const va = this._centroid(this._indices[left], centroidOffset);
        const vb = this._centroid(this._indices[middle], centroidOffset);
        const vc = this._centroid(this._indices[right], centroidOffset);

        if ((va <= vb && vb <= vc) || (vc <= vb && vb <= va)) {
            return vb;
        }
        if ((vb <= va && va <= vc) || (vc <= va && va <= vb)) {
            return va;
        }
        return vc;
    }

    private _swap(i: number, j: number): void {
        const tmp = this._indices[i];
        this._indices[i] = this._indices[j];
        this._indices[j] = tmp;
    }
}
