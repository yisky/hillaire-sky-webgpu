import { Color, MathUtils, Matrix4, NodeMaterial, Vector2, Vector3 } from 'three/webgpu'
import type { Camera, ColorRepresentation, Node, NodeBuilder, OrthographicCamera, PerspectiveCamera } from 'three/webgpu'
import { Fn, abs, clamp, dot, exp2, float, getViewPosition, inverseSqrt, length, logarithmicDepthToViewZ, max, min, mix, pow, saturate, select, time, uniform, uv, vec2, vec3, vec4, viewZToPerspectiveDepth } from 'three/tsl'
import { simplexNoise3D } from './simplexNoise3D'

// ln(2)。密度和透射率都以 2 为底，对 2^x 积分会出现 1 / ln(2) 系数
const LN2 = Math.LN2

// 高度差折算后的指数差小于该值时，按均匀密度处理，避免 ( 1 - 2^-x ) / x 在 x → 0 时出现 0 / 0
const UNIFORM_DENSITY_THRESHOLD = 0.01

// 指数上限。float32 的 2^x 在 x 约为 128 时溢出为 Inf，留出余量
const MAX_EXPONENT = 126.0

// 相机到像素距离平方的下限。小于它时视为与相机重合，直接输出原色
const MIN_DISTANCE_SQUARED = 1e-6

// viewZ 为负。只有视线深度进入远平面最后 0.1% 的像素才改噪声采样点
const FAR_VIEW_Z_LIMIT = 0.999

// 天空噪声沿雾面的最远距离，单位是噪声周期。采样坐标乘 noiseScale 后，一个周期约 1 / noiseScale 米
const NOISE_HORIZON_WAVELENGTHS = 2.0

// 视线几乎平行于坐标轴时避免除零
const RAY_AXIS_EPSILON = 1e-4

export interface SSFParams {
	enabled?: boolean
	fogColor?: ColorRepresentation
	fogMaxOpacity?: number
	fogDensity?: number
	fogHeight?: number
	fogHeightFalloff?: number
	fogStartDistance?: number
	noiseStrength?: number
	noiseScale?: number
	noiseSpeed?: Vector2
	inscatteringLightDirection?: Vector3
	directionalInscatteringColor?: ColorRepresentation
	directionalInscatteringExponent?: number
	directionalInscatteringStartDistance?: number
}

/**
 * 屏幕空间高度雾（Screen Space Fog）。
 *
 * 从场景深度重建每个像素的世界坐标，沿相机视线对指数高度雾密度做解析积分，再按透射率混合雾色、定向散射和场景颜色。
 * 计算写在 TSL 节点里，赋给 NodeMaterial.fragmentNode。RenderPipeline 用全屏四边形画这个材质。
 * 雾在线性空间里混合。色调映射不在这里做：管线默认用 renderOutput 包住 outputNode，内部创建 ToneMappingNode，在雾之后做 ACES。
 */
export class SSF {

	/** 全屏四边形使用的材质。fragmentNode 就是雾化后的颜色。 */
	readonly material = new NodeMaterial()

	/**
	 * 交给 RenderPipeline 的输出节点，和 material.fragmentNode 是同一个节点。
	 * 管线会再包一层 renderOutput，所以这里不要自己做色调映射。
	 */
	readonly outputNode: Node

	set enabled( value: boolean ) {

		this._enabled.value = value

	}

	get enabled(): boolean {

		return this._enabled.value

	}

	set fogColor( value: ColorRepresentation ) {

		this._fogColor.value.set( value )

	}

	set fogMaxOpacity( value: number ) {

		this._fogMaxOpacity.value = MathUtils.clamp( value, 0.0, 1.0 )

	}

	set fogDensity( value: number ) {

		this._fogDensity.value = value

	}

	set fogHeight( value: number ) {

		this._fogHeight.value = value

	}

	set fogHeightFalloff( value: number ) {

		this._fogHeightFalloff.value = value

	}

	set fogStartDistance( value: number ) {

		this._fogStartDistance.value = value

	}

	set noiseStrength( value: number ) {

		this._noiseStrength.value = value

	}

	set noiseScale( value: number ) {

		this._noiseScale.value = value

	}

	set noiseSpeed( value: Vector2 ) {

		this._noiseSpeed.value.copy( value )

	}

	set inscatteringLightDirection( value: Vector3 ) {

		this._inscatteringLightDirection.value.copy( value ).normalize()

	}

	set directionalInscatteringColor( value: ColorRepresentation ) {

		this._directionalInscatteringColor.value.set( value )

	}

	set directionalInscatteringExponent( value: number ) {

		this._directionalInscatteringExponent.value = value

	}

	set directionalInscatteringStartDistance( value: number ) {

		this._directionalInscatteringStartDistance.value = value

	}

	// 是否启用效果。关掉时按 0 混合，输出场景原色。用 float 而不是 select：select 在条件为假时会丢掉颜色采样，画面变黑
	private readonly _enabled = uniform( true )

	// 雾色。Color.set 会把 sRGB 转成线性工作空间，和场景颜色处在同一空间
	private readonly _fogColor = uniform( new Color() )

	// 雾的最大不透明度。透射率不会低于 1 - fogMaxOpacity
	private readonly _fogMaxOpacity = uniform( 1.0 )

	// 雾面高度处的密度
	private readonly _fogDensity = uniform( 0.02 )

	// 雾面高度（世界坐标 Y）
	private readonly _fogHeight = uniform( 10.0 )

	// 高度衰减系数。每升高 1 / value 个单位，密度减半。0 是均匀雾
	private readonly _fogHeightFalloff = uniform( 0.25 )

	// 雾效开始计算的距离。这段距离不进积分
	private readonly _fogStartDistance = uniform( 0.0 )

	// 噪声对雾面高度的最大扰动幅度
	private readonly _noiseStrength = uniform( 0.1 )

	// 噪声频率。越小团块越大
	private readonly _noiseScale = uniform( 0.01 )

	// 噪声沿世界 X、Z 轴的流动速度
	private readonly _noiseSpeed = uniform( new Vector2() )

	// 定向光传播方向（单位向量）。光从正上方射下时为 (0, -1, 0)，朝上看散射最强
	private readonly _inscatteringLightDirection = uniform( new Vector3() )

	// 定向散射颜色。演示里会先乘上强度，抵消 ACES 对光晕的压缩
	private readonly _directionalInscatteringColor = uniform( new Color() )

	// 定向散射聚焦指数。越大，光晕越窄
	private readonly _directionalInscatteringExponent = uniform( 8.0 )

	// 定向散射开始计算的距离
	private readonly _directionalInscatteringStartDistance = uniform( 0.0 )

	// 场景相机的世界矩阵。setCamera 换成相机自己的 matrixWorld，渲染时读到的就是最新值
	private readonly _cameraWorldMatrix = uniform( new Matrix4() )

	// 场景相机的投影逆矩阵。同样按引用绑定
	private readonly _cameraProjectionMatrixInverse = uniform( new Matrix4() )

	// 近、远裁剪面。只在开启对数深度缓冲时，用来把深度换回透视深度
	private readonly _cameraNear = uniform( 0.1 )
	private readonly _cameraFar = uniform( 1000.0 )

	constructor( colorNode: Node, depthNode: Node, params: SSFParams = {} ) {

		this.material.name = 'SSF'
		this.outputNode = this._initialize( colorNode, depthNode, params )
		this.material.fragmentNode = this.outputNode

	}

	/**
	 * 绑定场景相机。
	 *
	 * 全屏四边形用的是正交相机，TSL 内置的 cameraWorldMatrix、cameraPosition 指向那台相机。
	 * 这里必须用自己的 uniform 绑定场景相机。按引用赋值后，相机更新矩阵时 uniform 自动跟随，不用每帧拷贝。
	 */
	setCamera( camera: Camera ): void {

		this._cameraWorldMatrix.value = camera.matrixWorld
		this._cameraProjectionMatrixInverse.value = camera.projectionMatrixInverse

		const { near, far } = camera as PerspectiveCamera | OrthographicCamera
		this._cameraNear.value = near
		this._cameraFar.value = far

	}

	private _initialize( colorNode: Node, depthNode: Node, params: SSFParams ): Node {

		const {

			enabled = true,
			fogColor = '#bccbda',
			fogMaxOpacity = 1.0,
			fogDensity = 0.02,
			fogHeight = 10.0,
			fogHeightFalloff = 0.25,
			fogStartDistance = 0.0,
			noiseStrength = 0.1,
			noiseScale = 0.01,
			noiseSpeed = new Vector2( 0.01, 0.01 ),
			inscatteringLightDirection = new Vector3( 0.0, -1.0, 0.0 ),
			directionalInscatteringColor = '#fff3da',
			directionalInscatteringExponent = 8.0,
			directionalInscatteringStartDistance = 0.0

		} = params

		this.enabled = enabled
		this.fogColor = fogColor
		this.fogMaxOpacity = fogMaxOpacity
		this.fogDensity = fogDensity
		this.fogHeight = fogHeight
		this.fogHeightFalloff = fogHeightFalloff
		this.fogStartDistance = fogStartDistance
		this.noiseStrength = noiseStrength
		this.noiseScale = noiseScale
		this.noiseSpeed = noiseSpeed
		this.inscatteringLightDirection = inscatteringLightDirection
		this.directionalInscatteringColor = directionalInscatteringColor
		this.directionalInscatteringExponent = directionalInscatteringExponent
		this.directionalInscatteringStartDistance = directionalInscatteringStartDistance

		// 节点图只建一次。之后改的是 uniform，不重建材质
		return this._createOutputNode( colorNode, depthNode )

	}

	/**
	 * 计算一段射线的光学厚度。
	 *
	 * 密度为 D × 2^( -h × ( y - H ) )，射线高度从 y0 线性变化到 y1。
	 * 以两端中密度较大的一端为基准，避免绝对高程导致溢出或 Inf × 0。
	 *
	 * @returns 光学厚度 τ，透射率为 2^( -τ )
	 */
	private _getOpticalDepth( y0: any, y1: any, rayLength: any, fogHeight: any ): any {

		const { _fogDensity: density, _fogHeightFalloff: falloff } = this

		const startY = float( y0 )
		const endY = float( y1 )
		const distance = float( rayLength )
		const height = float( fogHeight )

		// 两端密度的以 2 为底的指数：density(y) = D × 2^e，e = -h × ( y - H )
		// 只用相对雾面的高度差。雾面设在几百、上千米也不会对世界坐标直接求 2^( -h × y )
		const e0 = falloff.negate().mul( startY.sub( height ) )
		const e1 = falloff.negate().mul( endY.sub( height ) )

		// 沿射线密度按 2^e 变化，积分结果为：
		//   τ = D × L × 2^eMax × ( 1 - 2^-x ) / ( x × ln2 )
		// 其中 x = | e0 - e1 |，eMax = max( e0, e1 )
		// 提出较大的 2^eMax 后，括号里的因子总在 ( 0, 1 ] 之间
		const x = abs( e0.sub( e1 ) )

		// x → 0 时 ( 1 - 2^-x ) / ( x × ln2 ) → 1，退化为均匀密度。衰减系数为 0 或视线水平时走这里
		// 另一分支的除数取 max( x, 阈值 )，即使被 select 丢弃也不会产生 0 / 0
		const lineIntegral = select(
			x.lessThan( UNIFORM_DENSITY_THRESHOLD ),
			float( 1.0 ),
			float( 1.0 ).sub( exp2( x.negate() ) ).div( max( x, UNIFORM_DENSITY_THRESHOLD ).mul( LN2 ) )
		)

		// 只限制指数上限：雾面下方很深时 τ 本来就极大，限制后透射率仍为 0
		// 指数下限不限制。相机远在雾面上方时 2^eMax 下溢为 0，正好表示没有雾
		const opticalDepth = density.mul( distance ).mul( exp2( min( max( e0, e1 ), MAX_EXPONENT ) ) ).mul( lineIntegral )

		return select( distance.greaterThan( 0.0 ), opticalDepth, float( 0.0 ) )

	}

	private _createOutputNode( colorNode: Node, depthNode: Node ): Node {

		const {

			_enabled: enabled,
			_fogColor: fogColor,
			_fogMaxOpacity: fogMaxOpacity,
			_fogHeight: fogHeight,
			_fogStartDistance: fogStartDistance,
			_noiseStrength: noiseStrength,
			_noiseScale: noiseScale,
			_noiseSpeed: noiseSpeed,
			_inscatteringLightDirection: inscatteringLightDirection,
			_directionalInscatteringColor: directionalInscatteringColor,
			_directionalInscatteringExponent: directionalInscatteringExponent,
			_directionalInscatteringStartDistance: directionalInscatteringStartDistance,
			_cameraWorldMatrix: cameraWorldMatrix,
			_cameraProjectionMatrixInverse: cameraProjectionMatrixInverse,
			_cameraNear: cameraNear,
			_cameraFar: cameraFar

		} = this

		const applyFog = Fn( ( [ inputColor ]: [ any ], builder: NodeBuilder ) => {

			// ---------- 1. 从深度重建世界坐标 ----------

			const screenUV = uv()

			// 深度缓冲中的原始深度。WebGPU 坐标系下范围是 [ 0, 1 ]
			let depth: any = ( depthNode as any ).sample( screenUV ).r

			// 开启对数深度缓冲时，深度缓冲存的是对数值，先换算回透视深度，getViewPosition 才能用投影逆矩阵还原
			// 反向深度缓冲不需要处理：渲染器会把反向深度写进相机投影矩阵，投影逆矩阵自然能还原
			// logarithmicDepthBuffer 在建图时读取，是编译期分支，不是每像素判断
			if ( builder.renderer.logarithmicDepthBuffer === true ) {

				const viewZ = logarithmicDepthToViewZ( depth, cameraNear, cameraFar )
				depth = viewZToPerspectiveDepth( viewZ, cameraNear, cameraFar )

			}

			// getViewPosition 会按渲染器坐标系处理 NDC 和 Y 轴翻转，并做透视除法。透视、正交相机都适用
			const viewPosition = getViewPosition( screenUV, depth, cameraProjectionMatrixInverse )

			// 观察空间 → 世界空间。相机世界坐标是世界矩阵的平移部分
			const worldPosition = cameraWorldMatrix.mul( vec4( viewPosition, 1.0 ) ).xyz
			const cameraPosition = cameraWorldMatrix.mul( vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz

			// ---------- 2. 视线方向与长度 ----------

			const cameraToReceiver = worldPosition.sub( cameraPosition )
			const cameraToReceiverLengthSquared = dot( cameraToReceiver, cameraToReceiver )

			// 取下限后再开方，像素与相机重合时也不会出现 inverseSqrt( 0 )。这种像素最后直接输出原色
			const cameraToReceiverLengthInv = inverseSqrt( max( cameraToReceiverLengthSquared, MIN_DISTANCE_SQUARED ) )
			const cameraToReceiverLength = cameraToReceiverLengthSquared.mul( cameraToReceiverLengthInv )
			const cameraToReceiverNormalized = cameraToReceiver.mul( cameraToReceiverLengthInv )

			// ---------- 3. 噪声扰动雾面高度 ----------

			// 地面和物体在命中点采样。天空没有表面，命中点落在远裁剪面上，相邻像素的世界坐标隔得太远，噪声会变成细斑
			// 观察空间的远平面是 z = -far，用 viewZ 比较。斜射线不会因为欧氏距离更长被当成天空
			// 天空改到视线与雾面的交点采样，水平位移限制在约两个噪声周期内
			const farViewZ = cameraFar.negate()
			const isFarPlane = viewPosition.z.lessThanEqual( farViewZ.mul( FAR_VIEW_Z_LIMIT ) )

			const horizontal = vec2( cameraToReceiverNormalized.x, cameraToReceiverNormalized.z )
			const horizontalLength = length( horizontal )
			const maxDistance = float( NOISE_HORIZON_WAVELENGTHS ).div( max( noiseScale, RAY_AXIS_EPSILON ) )

			// select 的未选中分支仍会执行，分母用 1 代替 0，结果会被丢掉
			const rayY = cameraToReceiverNormalized.y
			const safeRayY = select( abs( rayY ).greaterThan( RAY_AXIS_EPSILON ), rayY, float( 1.0 ) )
			const hit = fogHeight.sub( cameraPosition.y ).div( safeRayY )
			const hitInFront = abs( rayY ).greaterThan( RAY_AXIS_EPSILON ).and( hit.greaterThan( 0.0 ) )
			const distanceAlongRay = select( hitInFront, min( hit, maxDistance ), maxDistance )

			const hasHorizontal = horizontalLength.greaterThan( RAY_AXIS_EPSILON )
			const fogPlanePosition = vec3(
				cameraPosition.x.add( select( hasHorizontal, horizontal.x.mul( distanceAlongRay ), float( 0.0 ) ) ),
				fogHeight,
				cameraPosition.z.add( select( hasHorizontal, horizontal.y.mul( distanceAlongRay ), float( 0.0 ) ) )
			)
			const noisePosition = select( isFarPlane, fogPlanePosition, worldPosition )

			// time 是渲染器维护的全局时间（秒），不需要在 JS 侧每帧累加
			const noiseSamplePosition = noisePosition.mul( noiseScale ).add( vec3( time.mul( noiseSpeed.x ), 0.0, time.mul( noiseSpeed.y ) ) )

			// simplexNoise3D 输出 [ 0, 1 ]，映射回 [ -1, 1 ]，再乘以最大扰动幅度
			const noise = simplexNoise3D( noiseSamplePosition ).mul( 2.0 ).sub( 1.0 )
			const noisyFogHeight = fogHeight.add( noise.mul( noiseStrength ) )

			// ---------- 4. 起始距离：从排除段末端重新积分 ----------

			// 沿视线先走过起始距离（不超过视线全长），积分从这一点的高度开始，而不是按比例缩放整段结果
			// 密度随高度变化，只有重新积分才能表达「相机前 N 个单位不进雾」
			const fogExcludedLength = clamp( fogStartDistance, 0.0, cameraToReceiverLength )
			const dirExcludedLength = clamp( directionalInscatteringStartDistance, 0.0, cameraToReceiverLength )

			const fogStartY = cameraPosition.y.add( cameraToReceiverNormalized.y.mul( fogExcludedLength ) )
			const dirStartY = cameraPosition.y.add( cameraToReceiverNormalized.y.mul( dirExcludedLength ) )

			const fogOpticalDepth = this._getOpticalDepth( fogStartY, worldPosition.y, cameraToReceiverLength.sub( fogExcludedLength ), noisyFogHeight )
			const dirOpticalDepth = this._getOpticalDepth( dirStartY, worldPosition.y, cameraToReceiverLength.sub( dirExcludedLength ), noisyFogHeight )

			// ---------- 5. 定向散射 ----------

			// 视线反方向与光线传播方向的夹角余弦：迎着光看（视线指向光源）时为 1
			// 例如光从上往下照（传播方向 (0, -1, 0)），朝上看时 -视线 = (0, -1, 0)，余弦为 1，光晕最亮
			const lightAlignment = saturate( dot( cameraToReceiverNormalized.negate(), inscatteringLightDirection ) )
			const directionalLightInscattering = directionalInscatteringColor.mul( pow( lightAlignment, directionalInscatteringExponent ) )

			// 散射量与这段射线上被雾挡住的比例成正比：1 - 透射率
			const directionalInscattering = directionalLightInscattering.mul( float( 1.0 ).sub( saturate( exp2( dirOpticalDepth.negate() ) ) ) )

			// ---------- 6. 透射率与最终混合 ----------

			// 透射率 = 2^( -τ )，表示场景颜色穿过雾后剩下的比例。下限为 1 - 最大不透明度，保留远景细节
			const transmittance = max( saturate( exp2( fogOpticalDepth.negate() ) ), float( 1.0 ).sub( fogMaxOpacity ) )

			// 雾色按被遮挡比例叠加，场景颜色按透射率衰减，再加上定向散射
			const foggedColor = fogColor.mul( float( 1.0 ).sub( transmittance ) ).add( directionalInscattering ).add( inputColor.rgb.mul( transmittance ) )

			// 用 mix 而不是 select。select 会把未选中分支隔离掉，enabled 为 false 时场景颜色采样不再执行，输出变成黑色
			const applied = float( enabled.and( cameraToReceiverLengthSquared.greaterThan( MIN_DISTANCE_SQUARED ) ) )

			return vec4( mix( inputColor.rgb, foggedColor, applied ), inputColor.a )

		} )

		return applyFog( colorNode )

	}

}
