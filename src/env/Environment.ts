/**
 * 把天空画进环境贴图，供标准材质的漫反射和粗糙度反射使用。
 *
 * 只画一只内表面天空盒，不画建筑和树。太阳方向变化超过 0.004 弧度才重做。
 * 云不在这张图里，所以太阳不动就不用刷新。
 */

import {
	BackSide,
	BoxGeometry,
	Mesh,
	MeshBasicNodeMaterial,
	PMREMGenerator,
	Scene,
	Vector3,
	type RenderTarget,
	type WebGPURenderer,
} from 'three/webgpu';

import type { Sky } from '../sky';

const SIZE = 128;
const SUN_MOVE = 0.004;

export class SkyEnvironment {

	private readonly pmrem: PMREMGenerator;

	private readonly probe = new Scene();

	private readonly box: Mesh;

	private readonly lastSun = new Vector3();

	private target: RenderTarget | null = null;

	private ready = false;

	constructor(
		renderer: WebGPURenderer,
		private readonly scene: Scene,
		sky: Sky,
	) {

		const material = new MeshBasicNodeMaterial();
		material.side = BackSide;
		material.depthWrite = false;
		material.fog = false;
		material.colorNode = sky.reflectionNode;

		this.box = new Mesh( new BoxGeometry( 1, 1, 1 ), material );
		this.box.frustumCulled = false;
		this.box.scale.setScalar( 10 );
		this.probe.add( this.box );

		this.pmrem = new PMREMGenerator( renderer );

	}

	/** 在 `sky.update` 之后调用。查找表已经是这一帧的。 */
	update( sunDirection: Vector3 ): void {

		if ( this.ready && sunDirection.angleTo( this.lastSun ) <= SUN_MOVE ) return;

		this.lastSun.copy( sunDirection );
		this.ready = true;
		this.target = this.pmrem.fromScene( this.probe, 0, 0.1, 100, {
			size: SIZE,
			renderTarget: this.target,
		} );
		this.scene.environment = this.target.texture;

	}

	dispose(): void {

		this.box.geometry.dispose();
		( this.box.material as MeshBasicNodeMaterial ).dispose();
		this.target?.dispose();
		this.pmrem.dispose();

	}

}
