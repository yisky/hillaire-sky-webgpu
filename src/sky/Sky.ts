/**
 * 可加入场景的天空网格。
 *
 * 内表面盒对齐相机，`depthWrite` 关闭，因此不挡住场景物体。
 * 颜色是天空视图亮度，再叠加太阳盘、月亮和恒星。太阳盘不写入查找表。
 */

import {
	BackSide,
	BoxGeometry,
	Color,
	Mesh,
	StorageTexture,
	MeshBasicNodeMaterial,
	OrthographicCamera,
	PerspectiveCamera,
	Vector3,
	type Camera,
	type Node,
	type WebGPURenderer,
} from 'three/webgpu';
import { uniform } from 'three/tsl';

import {
	DEFAULT_COVERAGE,
	DEFAULT_GROUND_ALBEDO,
	DEFAULT_SUN_DIRECTION,
	GROUND_RADIUS_KM,
	SUN_ILLUMINANCE,
} from './constants';
import { createIrradiance } from './atmosphere/irradiance';
import { createMultiScatteringLut } from './atmosphere/multiScattering';
import { createSkyViewLut } from './atmosphere/skyView';
import { createTransmittanceLut } from './atmosphere/transmittance';
import { createReflectionSkyColor, createSkyColorNode, createSkyColorWithClouds } from './composite/background';
import { SkyProField } from './clouds/skypro';

/**
 * 构造天空时的可选项。未给出的项使用 {@link Sky} 上的默认值。
 */
export interface SkyOptions {

	/**
	 * 云量，范围 `0..1`。
	 *
	 * @default 0.49
	 */
	coverage?: number;

	/**
	 * 真太阳方向，世界空间、不必预先归一化。
	 *
	 * 大气始终按这个方向散射，即使它落到地平线以下。
	 * 传入的向量会被复制，之后请改 {@link Sky.sunDirection}，不要替换那个属性。
	 */
	sunDirection?: Vector3;

}

/**
 * 可放进场景的天空。
 *
 * 每个实例有自己的太阳方向、云量和大气参数。
 * 默认用法：
 *
 * ```ts
 * const sky = new Sky()
 * scene.add( sky )
 * sky.update( renderer, delta, camera )
 * renderer.render( scene, camera )
 * ```
 *
 * 网格是单位盒子，内表面朝向相机。它不会自己跟着相机，
 * 要在 `renderer.render` 之前调用 {@link Sky.update}，把中心移到相机并把尺寸收到远平面之内。
 * 这样宿主不用为天空加大 `camera.far`。
 *
 * 同一颜色节点也可以交给背景通道：`scene.backgroundNode = sky.node`。
 * 不要和 `scene.add( sky )` 同时用，否则天空会画两遍。两条路径都调用同一次 `update`。
 */
export class Sky extends Mesh {

	/**
	 * 天空颜色节点。
	 *
	 * 网格材质使用它。只走背景通道时，把它赋给 `scene.backgroundNode`。
	 * 打开云之后这里变成「晴空 × 透射 + 内散射」，关掉云再变回晴空。
	 * 请在切换云之后再读它。
	 */
	get node(): Node< 'vec3' > {

		return this.colorNode;

	}

	/**
	 * 透射查找表，`256×64`，`rgba16float`。
	 *
	 * 横轴从天顶到掠地，纵轴从地面到大气顶。纹素是三通道透射，1 为无衰减。
	 */
	readonly transmittance: StorageTexture;

	/**
	 * 多次散射查找表，`32×32`，`rgba16float`。
	 *
	 * 横轴是太阳天顶角余弦，左为 `-1`、右为 `1`。纵轴是高度，下为地面。
	 * 纹素是 `L / (1 - fms)`，还没乘太阳照度。
	 */
	readonly multiScattering: StorageTexture;

	/**
	 * 天空视图查找表，`192×108`，`rgba16float`。
	 *
	 * 横轴是相对太阳的方位，太阳在左侧。纵轴上半是天顶到地平线，下半是地平线以下。
	 * 纹素已乘场景照度，不含太阳盘。
	 */
	readonly skyView: StorageTexture;

	/**
	 * 真太阳方向，世界空间单位向量。
	 *
	 * 请原地修改（`.copy()` / `.set()`），不要给这个属性重新赋值。
	 * {@link Sky.sunDirectionNode} 读的是同一个向量。
	 */
	readonly sunDirection: Vector3;

	/**
	 * 太阳光的颜色，线性 RGB，是海平面透射，不含照度。
	 *
	 * 一开始是白。读回到达后换成透射的三个通道。亮度在 {@link Sky.sunIntensity}。
	 * {@link Sky.sunColorNode} 与这个 `Color` 是同一对象。
	 */
	readonly sunColor: Color;

	/**
	 * 太阳光强度，场景单位。
	 *
	 * 大气顶照度 `11` 乘地平线淡出：`smoothstep(sun.y, -0.03, 0.02)`。
	 * 太阳落到地平线下后趋近 0。{@link Sky.sunIntensityNode} 是同一个数。
	 */
	sunIntensity = SUN_ILLUMINANCE;

	/** 给 TSL 材质用的太阳强度。 */
	readonly sunIntensityNode = uniform( SUN_ILLUMINANCE );

	/**
	 * 天空对地面的辐照度除以 `π`，线性 RGB。存的是 `E/π`，不是 `E`。
	 *
	 * 初值是占位的淡蓝。积分读回后换成天空的 `E/π`，夜间再加一点环境。
	 * {@link Sky.skyIrradianceNode} 与这个 `Color` 是同一对象。
	 */
	readonly skyIrradiance: Color;

	/** 给其他 TSL 材质用的太阳颜色。不含照度。 */
	readonly sunColorNode: Node;

	/** 给其他 TSL 材质用的天空辐照度（`E/π`）。 */
	readonly skyIrradianceNode: Node;

	/**
	 * 云量，范围 `0..1`。
	 *
	 * SkyPro 用它做柱顶：越大，天气图上更多的地方够得着云顶。
	 * 云关闭时只保存数字，不分配纹理。
	 */
	coverage: number;

	/**
	 * 云的平移速度，米/秒。沿风向推动天气图和形状。
	 *
	 * SkyPro「Partly cloudy」的延时速度是 `89`。
	 *
	 * @default 89
	 */
	windSpeed: number;

	/**
	 * 形状噪声沿风向的滑动速度，米/秒。
	 *
	 * 和 {@link Sky.windSpeed} 的差是云体内部的翻滚。SkyPro 预设是 `60.8`。
	 *
	 * @default 60.8
	 */
	evolutionSpeed: number;

	/**
	 * 瑞利散射相对默认系数的缩放，无量纲。
	 *
	 * `1` 表示使用默认瑞利系数。
	 *
	 * @default 1
	 */
	rayleighScale: number;

	/**
	 * 米氏散射与消光的共同缩放，无量纲。
	 *
	 * @default 1
	 */
	mieScale: number;

	/**
	 * 米氏相位的不对称因子，范围 `(-1, 1)`。
	 *
	 * `0.8` 表示强前向散射。透射与方向无关，这个系数只进入相位函数。
	 *
	 * @default 0.8
	 */
	mieG: number;

	/**
	 * 臭氧吸收相对默认系数的缩放，无量纲。
	 *
	 * @default 1
	 */
	ozoneScale: number;

	/**
	 * 地面反照率，线性 RGB。进入多次散射查找表。
	 *
	 * 默认 `(0.06, 0.08, 0.1)`，比论文的 `0.3` 更暗。
	 */
	readonly groundAlbedo: Color;

	/**
	 * 本实例派发 compute 使用的渲染器，由 {@link Sky.update} 写入。
	 */
	private renderer: WebGPURenderer | null;

	/**
	 * 本帧步长，单位秒，范围 `0..0.1`。
	 *
	 * 负时间没有意义。大于 `0.1` 秒的部分丢掉，避免一帧积分跨过太长的时间。
	 */
	private frameDelta = 0;

	private readonly transmittanceLut = createTransmittanceLut();

	/**
	 * 上次写入透射表时的三个缩放。
	 *
	 * 地面反照率不进入透射。缩放变了才重写透射，并连带重写多次散射。
	 */
	private builtScales: { rayleigh: number; mie: number; ozone: number } | null = null;

	/**
	 * 上次写入多次散射表时的地面反照率。
	 */
	private builtAlbedo: { r: number; g: number; b: number } | null = null;

	private readonly multiScatteringLut: ReturnType< typeof createMultiScatteringLut >;

	private readonly skyViewLut: ReturnType< typeof createSkyViewLut >;

	/**
	 * 上次写入天空视图时的太阳方向、量化高度和 `mieG`。
	 *
	 * 高度单位是到地心的 km。相机只转不升降时这些数不变，表不重算。
	 */
	private builtSky: { x: number; y: number; z: number; height: number; mieG: number } | null = null;

	/**
	 * 月亮方向，世界空间单位向量。
	 *
	 * 请原地修改（`.copy()` / `.set()`），不要给这个属性重新赋值。
	 * 方向由宿主在 `update` 之前写入。初值朝向天顶。
	 * {@link Sky.moonDirectionNode} 是同一个向量。
	 */
	readonly moonDirection = new Vector3( 0, 1, 0 );

	/** 月亮方向节点，与 {@link Sky.moonDirection} 是同一个向量。 */
	readonly moonDirectionNode = uniform( this.moonDirection );

	/**
	 * 月光颜色，线性 RGB。
	 *
	 * 固定为 `(0.6, 0.7, 1)`，偏冷白。亮度在 {@link Sky.moonIntensity}。
	 * {@link Sky.moonColorNode} 与这个 `Color` 是同一对象。
	 */
	readonly moonColor = new Color( 0.6, 0.7, 1 );

	/** 给其他 TSL 材质用的月光颜色。 */
	readonly moonColorNode = uniform( this.moonColor );

	/**
	 * 月光强度，场景单位。
	 *
	 * `0.12 × 夜深`。夜深是 `smoothstep(-sun.y, 0.02, 0.18)`：
	 * 太阳还在地平线附近时为 0，降到约 `-0.18` 时为 1。
	 * {@link Sky.moonIntensityNode} 是同一个数。
	 */
	moonIntensity = 0;

	/** 给 TSL 材质用的月光强度。 */
	readonly moonIntensityNode = uniform( 0 );

	/**
	 * 夜深，`0` 到 `1`。太阳高度从 `-0.02` 降到 `-0.18` 时平滑打开，恒星和月亮用它。
	 */
	private readonly starIntensityNode = uniform( 0 );

	/** 累计时间，单位秒。恒星闪烁用它，不参与查找表。 */
	private readonly timeNode = uniform( 0 );

	private time = 0;

	private readonly irradiance: ReturnType< typeof createIrradiance >;

	/** 距下次辐照积分的秒数。到 0 才派发，读回未完成时不再派。 */
	private irradianceTimer = 0;

	/** 已经派发、还在等 `getArrayBufferAsync` 的结果。 */
	private irradiancePending = false;

	private disposed = false;

	/**
	 * 体积云。外面创建后赋到这里，天空只把它画进背景。
	 *
	 * 没赋就是晴空。天空不创建也不释放这一份。
	 */
	clouds: SkyProField | null = null;

	/** 当前交给材质和 `node` 的颜色。晴空和全景云各一份。 */
	private colorNode: Node< 'vec3' >;

	/** 不含云的颜色。关掉云时换回它。 */
	private clearNode: Node< 'vec3' >;

	/**
	 * 环境贴图用的天空。没有太阳盘、月亮盘和视锥云，恒星更暗。
	 * 和屏幕上的天空分开，避免海色和云被烤进反射。
	 */
	readonly reflectionNode: Node< 'vec3' >;

	/** 叠了全景云的颜色。云关掉后丢掉，下次打开再造。 */
	private cloudyNode: Node< 'vec3' > | null = null;

	/**
	 * 真太阳方向的 TSL 节点，与 {@link Sky.sunDirection} 是同一个向量。
	 *
	 * 大气散射和太阳光都用它。月光用 {@link Sky.moonDirectionNode}。
	 * 请继续改 `sunDirection`，不要改这个节点里的对象。
	 */
	readonly sunDirectionNode: Node;

	/**
	 * @param options 云量、初始太阳方向
	 */
	constructor( options: SkyOptions = {} ) {

		const geometry = new BoxGeometry( 1, 1, 1 );
		const material = new MeshBasicNodeMaterial();

		super( geometry, material );

		material.side = BackSide;
		material.depthWrite = false;
		material.fog = false;
		this.name = 'Sky';
		this.frustumCulled = false;
		this.castShadow = false;
		this.receiveShadow = false;
		this.renderOrder = - 1;

		this.sunDirection = ( options.sunDirection ?? DEFAULT_SUN_DIRECTION ).clone().normalize();
		this.sunColor = new Color( 1, 1, 1 );
		this.skyIrradiance = new Color( 0.3, 0.4, 0.6 );
		this.sunColorNode = uniform( this.sunColor );
		this.skyIrradianceNode = uniform( this.skyIrradiance );
		this.coverage = options.coverage ?? DEFAULT_COVERAGE;
		this.windSpeed = 89;
		this.evolutionSpeed = 60.8;
		this.rayleighScale = 1;
		this.mieScale = 1;
		this.mieG = 0.8;
		this.ozoneScale = 1;
		this.groundAlbedo = DEFAULT_GROUND_ALBEDO.clone();
		this.renderer = null;
		this.sunDirectionNode = uniform( this.sunDirection );
		this.transmittance = this.transmittanceLut.texture;
		this.multiScatteringLut = createMultiScatteringLut(
			this.transmittance,
			this.transmittanceLut.rayleighScale,
			this.transmittanceLut.mieScale,
			this.transmittanceLut.ozoneScale,
			this.groundAlbedo,
		);
		this.multiScattering = this.multiScatteringLut.texture;
		this.skyViewLut = createSkyViewLut(
			this.transmittance,
			this.multiScattering,
			this.transmittanceLut.rayleighScale,
			this.transmittanceLut.mieScale,
			this.transmittanceLut.ozoneScale,
			this.sunDirectionNode,
		);
		this.skyView = this.skyViewLut.texture;
		this.irradiance = createIrradiance(
			this.skyView,
			this.transmittance,
			this.skyViewLut.viewHeight,
			this.sunDirectionNode,
		);
		this.clearNode = createSkyColorNode(
			this.skyView,
			this.transmittance,
			this.skyViewLut.viewHeight,
			this.sunDirectionNode,
			this.moonDirectionNode,
			this.starIntensityNode,
			this.timeNode,
		);
		this.colorNode = this.clearNode;
		this.reflectionNode = createReflectionSkyColor(
			this.skyView,
			this.skyViewLut.viewHeight,
			this.sunDirectionNode,
			this.moonDirectionNode,
			this.starIntensityNode,
			this.timeNode,
		);
		material.colorNode = this.colorNode;

	}

	/**
	 * 推进一帧天空。在 `renderer.render` 之前调用。
	 *
	 * 天空盒中心对齐相机。单位盒子覆盖 `[-0.5, 0.5]`，缩放取远平面的 `0.8`，
	 * 半边长就是远平面的 40%：近于远裁剪面，又远于近裁剪面。
	 * 写下位置和缩放后立刻 `updateMatrixWorld`，本帧读取世界矩阵时已经是新变换。
	 * 瑞利、米氏或臭氧缩放变化时先重写透射表，再重写多次散射和天空视图。
	 * 只改地面反照率时透射表保持不动。`mieG`、太阳高度和量化后的相机高度只重写天空视图。
	 * 相机原地转动不改变这些数，天空视图表不重算。
	 *
	 * @param renderer 本实例派发 compute 使用的渲染器
	 * @param delta 距上一帧的秒数。负值当作 `0`，大于 `0.1` 的部分丢掉
	 * @param camera 当前相机。`position.y` 为海拔，单位米
	 */
	update( renderer: WebGPURenderer, delta: number, camera: Camera ): void {

		this.renderer = renderer;
		this.frameDelta = Math.min( Math.max( delta, 0 ), 0.1 );
		this.position.copy( camera.position );
		this.scale.setScalar( readCameraFar( camera ) * 0.8 );
		this.updateMatrixWorld();
		this.writeCelestial();
		this.writeLookups( renderer, camera );
		this.integrateIrradiance( renderer );
		this.syncClouds( renderer, camera );

	}

	/**
	 * 外面赋了 {@link Sky.clouds} 就推进那一份，没赋就画晴空。
	 *
	 * @param renderer 本实例派发 compute 使用的渲染器
	 * @param camera 当前相机
	 */
	private syncClouds( renderer: WebGPURenderer, camera: Camera ): void {

		const field = this.clouds;

		if ( ! field ) {

			if ( this.cloudyNode ) this.useCloudColor( false );

			return;

		}

		if ( ! this.cloudyNode ) this.useCloudColor( true );

		field.update(
			renderer,
			this.coverage,
			camera,
			this.frameDelta,
			this.sunDirection,
			this.starIntensityNode.value as number,
			this.skyIrradiance,
			this.skyViewLut.viewHeight.value as number,
			this.windSpeed,
			this.evolutionSpeed,
		);

	}

	/**
	 * 切换材质和 {@link Sky.node} 用的颜色。
	 *
	 * 打开时用云历史合成，关掉时回到晴空。背景通道如果早就拿走了旧节点，需要再读一次 `node`。
	 *
	 * @param enabled 为真时叠上云
	 */
	private useCloudColor( enabled: boolean ): void {

		if ( enabled && this.clouds ) {

			this.cloudyNode = createSkyColorWithClouds(
				this.skyView,
				this.transmittance,
				this.skyViewLut.viewHeight,
				this.sunDirectionNode,
				this.moonDirectionNode,
				this.starIntensityNode,
				this.timeNode,
				this.clouds.viewClouds,
				this.clouds.viewCamera,
			);
			this.colorNode = this.cloudyNode;

		} else {

			this.cloudyNode = null;
			this.colorNode = this.clearNode;

		}

		const material = this.material as MeshBasicNodeMaterial;

		material.colorNode = this.colorNode;
		material.needsUpdate = true;

	}

	/**
	 * 由太阳高度写夜深，并累计闪烁用的时间。
	 *
	 * 太阳强度随高度淡出，月光强度随夜深升起，两者分开。月亮方向由宿主写入。
	 */
	private writeCelestial(): void {

		const sun = this.sunDirection;
		const night = smoothstep( - sun.y, 0.02, 0.18 );

		this.starIntensityNode.value = night;
		this.sunIntensity = SUN_ILLUMINANCE * smoothstep( sun.y, - 0.03, 0.02 );
		this.sunIntensityNode.value = this.sunIntensity;
		this.moonIntensity = 0.12 * night;
		this.moonIntensityNode.value = this.moonIntensity;
		this.time += this.frameDelta;
		this.timeNode.value = this.time;

	}

	/**
	 * 每 0.25 秒积分一次天空辐照，并异步读回。
	 *
	 * 上一次读回还没回来时不派发。结果晚一帧写进 {@link Sky.sunColor} 和 {@link Sky.skyIrradiance}。
	 * `sunColor` 只存海平面透射，不乘照度，也不换成月光。
	 * 辐照度是 `E/π`，夜间再加 `0.012 × 夜深` 的一点环境。
	 *
	 * @param renderer 派发 compute 并读回缓冲的渲染器
	 */
	private integrateIrradiance( renderer: WebGPURenderer ): void {

		this.irradianceTimer -= this.frameDelta;

		if ( this.irradianceTimer > 0 || this.irradiancePending ) return;

		this.irradianceTimer = 0.25;
		this.irradiancePending = true;
		renderer.compute( this.irradiance.compute );
		renderer.getArrayBufferAsync( this.irradiance.attribute ).then( ( buffer ) => {

			this.irradiancePending = false;

			if ( this.disposed ) return;

			const values = new Float32Array( buffer );
			const night = smoothstep( - this.sunDirection.y, 0.02, 0.18 );

			this.sunColor.setRGB( values[ 4 ], values[ 5 ], values[ 6 ] );

			const ambient = 0.012 * night;

			this.skyIrradiance.setRGB(
				values[ 0 ] + ambient * 0.6,
				values[ 1 ] + ambient * 0.7,
				values[ 2 ] + ambient,
			);

		} ).catch( () => {

			this.irradiancePending = false;

		} );

	}

	/**
	 * 按当前散射缩放和地面反照率写入查找表。
	 *
	 * 透射只依赖三个缩放。多次散射还依赖地面反照率，并读取透射表。
	 * 天空视图读取前两张表，所以三次 compute 按这个顺序派发。
	 * 相机高度先按近处 2 m、高处 2% 量化，再换成到地心的千米。
	 *
	 * @param renderer 派发 compute 的渲染器
	 * @param camera 当前相机。`position.y` 为海拔，单位米
	 */
	private writeLookups( renderer: WebGPURenderer, camera: Camera ): void {

		const built = this.builtScales;
		const scalesChanged = built === null
			|| built.rayleigh !== this.rayleighScale
			|| built.mie !== this.mieScale
			|| built.ozone !== this.ozoneScale;
		const albedo = this.groundAlbedo;
		const builtAlbedo = this.builtAlbedo;
		const albedoChanged = builtAlbedo === null
			|| builtAlbedo.r !== albedo.r
			|| builtAlbedo.g !== albedo.g
			|| builtAlbedo.b !== albedo.b;

		const height = quantizedViewHeight( camera.position.y );
		const sun = this.sunDirection;
		const builtSky = this.builtSky;
		const skyChanged = builtSky === null
			|| builtSky.x !== sun.x
			|| builtSky.y !== sun.y
			|| builtSky.z !== sun.z
			|| builtSky.height !== height
			|| builtSky.mieG !== this.mieG;

		if ( scalesChanged === false && albedoChanged === false && skyChanged === false ) return;

		if ( scalesChanged ) {

			this.transmittanceLut.rayleighScale.value = this.rayleighScale;
			this.transmittanceLut.mieScale.value = this.mieScale;
			this.transmittanceLut.ozoneScale.value = this.ozoneScale;
			renderer.compute( this.transmittanceLut.compute );
			this.builtScales = {
				rayleigh: this.rayleighScale,
				mie: this.mieScale,
				ozone: this.ozoneScale,
			};

		}

		if ( scalesChanged || albedoChanged ) {

			renderer.compute( this.multiScatteringLut.compute );
			this.builtAlbedo = { r: albedo.r, g: albedo.g, b: albedo.b };

		}

		if ( skyChanged || scalesChanged || albedoChanged ) {

			this.skyViewLut.viewHeight.value = height;
			this.skyViewLut.mieG.value = this.mieG;
			renderer.compute( this.skyViewLut.compute );
			this.builtSky = { x: sun.x, y: sun.y, z: sun.z, height, mieG: this.mieG };

		}

	}

	/**
	 * 释放这个实例的几何体、材质和查找表。
	 */
	override dispose(): void {

		this.disposed = true;
		this.geometry.dispose();
		const material = this.material;

		if ( Array.isArray( material ) ) {

			for ( const entry of material ) entry.dispose();

		} else {

			material.dispose();

		}

		this.transmittanceLut.compute.dispose();
		this.multiScatteringLut.compute.dispose();
		this.skyViewLut.compute.dispose();
		this.irradiance.compute.dispose();
		this.transmittance.dispose();
		this.multiScattering.dispose();
		this.skyView.dispose();
		releaseIrradianceBuffer( this.renderer, this.irradiance.attribute );

	}

}

/**
 * 读取透视或正交相机的远平面。
 *
 * `Camera` 基类没有 `far`。天空盒半边长取这个距离的 40%，
 * 单位与场景一致（演示里是米）。
 *
 * @param camera 正在渲染的相机
 * @returns 远平面距离。其他相机类型退回 `100`
 */
function readCameraFar( camera: Camera ): number {

	if ( camera instanceof PerspectiveCamera || camera instanceof OrthographicCamera ) {

		return camera.far;

	}

	return 100;

}

/**
 * 丢掉这个实例的辐照存储缓冲。
 *
 * 渲染器在创建存储缓冲时没有接上 `BufferAttribute.dispose()`，
 * 属性表的 `delete` 才会销毁 GPU 缓冲并减掉计数。
 *
 * @param renderer 创建过这块缓冲的渲染器。还没 `update` 过则为空
 * @param attribute 辐照积分写入的缓冲
 */
function releaseIrradianceBuffer( renderer: WebGPURenderer | null, attribute: { dispose(): void } ): void {

	const attributes = ( renderer as WebGPURenderer & { _attributes: { delete( attribute: unknown ): unknown } | null } | null )?._attributes;

	attributes?.delete( attribute );
	attribute.dispose();

}

/**
 * 把相机海拔量化成天空视图使用的地心距离。
 *
 * 低于 100 m 时按 2 m 取整，再高按 2%（`log(1.02)`）取整，避免身体起伏就重算整张表。
 * 输入的 0.5 m 偏移让观察点落在量化格点上。结果加上地面半径，单位 km。
 *
 * @param altitudeMeters 相机世界坐标的 y，单位米
 * @returns 到地心的距离，单位 km
 */
function quantizedViewHeight( altitudeMeters: number ): number {

	const altitude = Math.max( 0.5, altitudeMeters + 0.5 );
	const quantized = altitude < 100
		? Math.round( altitude / 2 ) * 2
		: Math.exp( Math.round( Math.log( altitude ) / Math.log( 1.02 ) ) * Math.log( 1.02 ) );

	return GROUND_RADIUS_KM + Math.max( 0.001, quantized / 1000 );

}

/**
 * Hermite 平滑阶跃。
 *
 * `x <= edge0` 为 0，`x >= edge1` 为 1，中间是 `t²(3-2t)`。
 *
 * @param x 输入
 * @param edge0 下沿
 * @param edge1 上沿
 * @returns `0..1`
 */
function smoothstep( x: number, edge0: number, edge1: number ): number {

	const t = Math.min( Math.max( ( x - edge0 ) / ( edge1 - edge0 ), 0 ), 1 );

	return t * t * ( 3 - 2 * t );

}
