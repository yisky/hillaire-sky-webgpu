/**
 * 把天空烤进环境贴图，给标准材质做漫反射和镜面反射。
 *
 * 探针场景里只有一只内表面盒子，不画建筑和树。颜色走 {@link createReflectionSkyColor}：
 * 大气和月光夜空保留，没有太阳盘、月亮盘，也没有视锥云。恒星再乘 `0.08`，
 * 避免亮星在粗糙反射里闪。真正的太阳光和月光由场景里的灯出，不靠这张图里的亮点。
 * 云是相机视锥的历史缓冲，六面体上看不到同一块，所以也不烤进来。
 *
 * 不是后期节点，不进 `pipeline.outputNode`。`sky.update` 写完查找表之后再调 {@link SkyProbe.update}，
 * 太阳方向变化超过 {@link SUN_MOVE} 才重烤。结果写到主场景的 `environment`。
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
import { createReflectionSkyColor } from '../sky/composite/background';

/** 立方体贴图边长。够玻璃和金属反射天空，不必跟画面一样大。 */
const SIZE = 128;

/** 太阳转过这么多弧度才重烤。约 `0.23°`，避免每帧卷积。 */
const SUN_MOVE = 0.004;

/**
 * 天空环境探针。
 *
 * 构造时传入渲染器、主场景和天空。主场景用来挂 `environment`，天空只读查找表和方向节点。
 * 每帧在 `sky.update` 之后把太阳方向传进来；第一次一定烤，之后方向没动就跳过。
 */
export class SkyProbe {

	/** 把盒子场景收成预滤波环境贴图。 */
	private readonly pmrem: PMREMGenerator;

	/** 只放天空盒子，PMREM 从这里拍六面。 */
	private readonly probe = new Scene();

	/** 内表面盒子。尺度 `10`，落在近 `0.1`、远 `100` 的立方体相机里。 */
	private readonly box: Mesh;

	/** 上一次烤的时候太阳朝哪。 */
	private readonly lastSun = new Vector3();

	/** 当前这张预滤波图。重烤时交给 `fromScene` 复用，免得每帧新建。 */
	private target: RenderTarget | null = null;

	/** 已经烤过至少一次。第一次不能靠夹角判断，否则初始方向也会被跳过。 */
	private ready = false;

	/**
	 * @param renderer 当前渲染器，PMREM 用它画立方体
	 * @param scene 主场景，烤完把 `environment` 指到这张图
	 * @param sky 已经建好的天空。盒子读它的视图表、高度和日月恒星节点
	 */
	constructor(
		renderer: WebGPURenderer,
		private readonly scene: Scene,
		sky: Sky,
	) {

		const material = new MeshBasicNodeMaterial();
		material.side = BackSide;
		material.depthWrite = false;
		material.fog = false;
		material.colorNode = createReflectionSkyColor(
			sky.skyView,
			sky.viewHeight,
			sky.sunDirectionNode,
			sky.moonDirectionNode,
			sky.starIntensityNode,
			sky.timeNode,
		);

		this.box = new Mesh( new BoxGeometry( 1, 1, 1 ), material );
		this.box.frustumCulled = false;
		this.box.scale.setScalar( 10 );
		this.probe.add( this.box );

		this.pmrem = new PMREMGenerator( renderer );

	}

	/**
	 * 太阳动得够多就重烤，写到主场景的 `environment`。
	 *
	 * 必须在 `sky.update` 之后调：视图表、恒星强度这一帧才是对的。
	 * 盒子材质绑的是天空的节点，不用再抄一份方向。
	 *
	 * @param sunDirection 这一帧的真太阳方向
	 */
	update( sunDirection: Vector3 ): void {

		if ( this.ready && sunDirection.angleTo( this.lastSun ) <= SUN_MOVE ) return;

		this.lastSun.copy( sunDirection );
		this.ready = true;
		// 不模糊。近远包住盒子。已有目标就复用。
		this.target = this.pmrem.fromScene( this.probe, 0, 0.1, 100, {
			size: SIZE,
			renderTarget: this.target,
		} );
		this.scene.environment = this.target.texture;

	}

	/** 盒子、贴图和生成器。主场景的 `environment` 调用方自己清。 */
	dispose(): void {

		this.box.geometry.dispose();
		( this.box.material as MeshBasicNodeMaterial ).dispose();
		this.target?.dispose();
		this.pmrem.dispose();

	}

}
