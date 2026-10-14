/**
 * 太阳的镜头炫光。可见度读场景深度，有云时再读主视图云。
 *
 * 没有贴图。七叶片光圈的鬼影沿太阳和画面中心的连线排开，画面中心有一圈色散光环，
 * 太阳周围是衍射星芒、细纹和一层把暗部抬起来的眩光。
 * 可见度在 GPU 上量一次：太阳圆盘上 24 个点，深度仍是清除值的算露出来，
 * 再乘云的透射、太阳高度和是否在画内。结果按 `1 - exp(-14 dt)` 跟上。
 * 深度清除值是 `1`。反向深度下天空是 `0`，这里按正向深度反过来比。
 * 太阳落下后炫光跟着灭，不接月亮。
 */

import {
	Color,
	PerspectiveCamera,
	StorageBufferAttribute,
	Vector2,
	Vector3,
	type ComputeNode,
	type Node,
	type Texture,
	type WebGPURenderer,
} from 'three/webgpu';
import {
	Fn,
	abs,
	atan,
	clamp,
	cos,
	exp,
	float,
	floor,
	fract,
	int,
	ivec2,
	length,
	max,
	mix,
	pow,
	screenUV,
	sin,
	smoothstep,
	storage,
	texture,
	textureLoad,
	textureSize,
	vec2,
	vec3,
	vec4,
	uniform,
} from 'three/tsl';
import { sampleSkyPro, type SkyProField } from '../sky/clouds/skypro';

const BLADES = 7;
const ROT = 0.3;
const SEG = ( 2 * Math.PI ) / BLADES;
const SPIKE = Math.PI / BLADES;
const TAPS = 24;

const GHOSTS = [
	{ a: 0.72, r: 0.018, tint: [ 1, 0.85, 0.6 ], k: 0.55 },
	{ a: 0.44, r: 0.045, tint: [ 0.55, 0.9, 1 ], k: 0.35 },
	{ a: 0.16, r: 0.022, tint: [ 0.8, 1, 0.7 ], k: 0.45 },
	{ a: - 0.18, r: 0.07, tint: [ 0.6, 0.75, 1 ], k: 0.22 },
	{ a: - 0.42, r: 0.03, tint: [ 1, 0.7, 0.9 ], k: 0.4 },
	{ a: - 0.75, r: 0.11, tint: [ 0.7, 1, 0.85 ], k: 0.14 },
	{ a: - 1.15, r: 0.05, tint: [ 1, 0.8, 0.55 ], k: 0.28 },
];

const sunPoint = new Vector3();
const sunNdc = new Vector3();
const forward = new Vector3();
const bufferSize = new Vector2();

/**
 * 构造炫光时的输入。太阳方向、颜色、亮度是实例上的 uniform，构造之后写 `.value`。
 */
export interface GlareOptions {

	/** 场景通道的深度纹理。清除值 `1` 表示这一像素没有物体。 */
	depth: Texture;

	/** 外面持有的体积云。没有时不乘透射。 */
	clouds: SkyProField | null;

}

/**
 * 镜头炫光。
 *
 * 每帧先 {@link Glare.update}，再画后期，最后 `renderer.compute(glare.visibility)`。
 * 可见度晚一帧：这一帧的深度要等场景画完才能读。
 */
export class Glare {

	/** 可见度。场景画完后派发，下一帧的炫光才读到新值。 */
	readonly visibility: ComputeNode;

	/** 总强度。默认 `1`。 */
	readonly strength = uniform( 1 );

	/** 指向太阳的世界方向。每帧写 `.value`。 */
	readonly sunDirection = uniform( new Vector3( 0, 1, 0 ) );

	/** 太阳颜色。每帧写 `.value`。 */
	readonly sunColor = uniform( new Color( 1, 1, 1 ) );

	/** 太阳亮度。落到地平线下后趋近 `0`，炫光跟着灭。每帧写 `.value`。 */
	readonly sunIntensity = uniform( 0 );

	private readonly depth: Texture;

	private readonly clouds: GlareOptions[ 'clouds' ];

	private readonly sunUV = uniform( new Vector2( 0.5, 0.5 ) );

	private readonly inView = uniform( 0 );

	private readonly aspect = uniform( 1 );

	private readonly discPx = uniform( 6 );

	private readonly dt = uniform( 1 / 60 );

	private readonly outputHeight = uniform( 1080 );

	private readonly flare: Node< 'vec3' >;

	/**
	 * @param options 场景深度和云
	 */
	constructor( options: GlareOptions ) {

		this.depth = options.depth;
		this.clouds = options.clouds;

		const attribute = new StorageBufferAttribute( 1, 1 );
		const written = storage( attribute, 'float', 1 );
		const read = storage( attribute, 'float', 1 ).toReadOnly();

		this.visibility = this.createVisibility( written.element( 0 ) as Node< 'float' > );
		this.flare = this.createFlare( read.element( 0 ) as Node< 'float' > );

	}

	/**
	 * 把炫光加到场景颜色上。
	 *
	 * @param color 场景通道的颜色
	 */
	composite( color: Node< 'vec4' > ): Node< 'vec4' > {

		return color.add( vec4( this.flare, 0 ) );

	}

	/**
	 * 按当前 {@link Glare.sunDirection} 写下太阳的屏幕位置和圆盘像素半径。
	 * 调用前先写好 `sunDirection`、`sunColor`、`sunIntensity` 的 `.value`。
	 * 只接透视相机：圆盘大小用 `fov`。
	 *
	 * @param renderer 用来读绘制缓冲高度
	 * @param camera 当前透视相机
	 * @param delta 距上一帧的秒数
	 */
	update( renderer: WebGPURenderer, camera: PerspectiveCamera, delta: number ): void {

		const sun = this.sunDirection.value;
		renderer.getDrawingBufferSize( bufferSize );
		sunPoint.copy( camera.position ).addScaledVector( sun, 1000 );
		sunNdc.copy( sunPoint ).project( camera );
		forward.set( 0, 0, - 1 ).applyQuaternion( camera.quaternion );

		const ahead = forward.dot( sun ) > 0.05;
		const edge = Math.max( Math.abs( sunNdc.x ), Math.abs( sunNdc.y ) );

		this.sunUV.value.set( sunNdc.x * 0.5 + 0.5, 0.5 - sunNdc.y * 0.5 );
		this.inView.value = ahead ? Math.min( Math.max( ( 1 - edge ) / 0.1, 0 ), 1 ) : 0;
		this.aspect.value = camera.aspect;
		this.dt.value = Math.min( Math.max( delta, 0 ), 0.1 );
		this.outputHeight.value = bufferSize.y;
		this.discPx.value = Math.max( 1.5, 0.004625 / Math.tan( camera.fov * Math.PI / 360 ) * bufferSize.y * 0.5 );

	}

	/**
	 * 量太阳圆盘有多少露在外面。
	 *
	 * @param current 可写的可见度
	 */
	private createVisibility( current: Node< 'float' > ): ComputeNode {

		const depth = texture( this.depth );
		const sunUV = this.sunUV;
		const inView = this.inView;
		const discPx = this.discPx;
		const dt = this.dt;

		return Fn( () => {

			const size = textureSize( depth, int( 0 ) ) as unknown as Node< 'vec2' >;
			const center = sunUV.mul( size );
			let sky: Node< 'float' > = float( 0 );

			for ( let i = 0; i < TAPS; i ++ ) {

				const radius = Math.sqrt( ( i + 0.5 ) / TAPS );
				const angle = i * 2.39996323;
				const point = center.add( vec2( Math.cos( angle ) * radius, Math.sin( angle ) * radius ).mul( discPx ) );
				const coord = ivec2( clamp( point, vec2( 0, 0 ), size.sub( 1 ) ) );
				const depthSample = textureLoad( depth, coord, int( 0 ) ) as unknown as Node< 'float' >;

				sky = sky.add( depthSample.greaterThan( 0.999 ).select( float( 1 ), float( 0 ) ) );

			}

			const up = smoothstep( - 0.02, 0.04, this.sunDirection.y );
			const cloudT = this.cloudTransmittance( this.sunDirection as Node< 'vec3' > );
			const target = sky.div( TAPS ).mul( cloudT ).mul( up ).mul( inView );

			current.assign( mix( current, target, float( 1 ).sub( exp( dt.mul( - 14 ) ) ) ) );

		} )().compute( 1 ) as ComputeNode;

	}

	/**
	 * 这一像素要加上的炫光。可见度为 `0` 时结果是黑。
	 *
	 * @param vis 只读的可见度
	 */
	private createFlare( vis: Node< 'float' > ): Node< 'vec3' > {

		const sunUV = this.sunUV;
		const aspect = this.aspect;
		const outputHeight = this.outputHeight;
		const strength = this.strength;

		return Fn( () => {

			const axes = vec2( aspect, 1 );
			const flipY = vec2( 1, - 1 );
			const p = screenUV.sub( 0.5 ).mul( axes ).mul( flipY );
			const s = sunUV.sub( 0.5 ).mul( axes ).mul( flipY );
			const key = this.sunColor.mul( this.sunIntensity );
			const light = key.mul( vis ).mul( strength ).mul( 0.02 );

			let ghosts: Node< 'vec3' > = vec3( 0, 0, 0 );

			for ( const ghost of GHOSTS ) {

				const dist = apertureDistance( p.sub( s.mul( ghost.a ) ) );
				const body = vec3(
					smoothstep( ghost.r * 0.975, ghost.r * 0.975 * 0.9, dist ),
					smoothstep( ghost.r, ghost.r * 0.9, dist ),
					smoothstep( ghost.r * 1.025, ghost.r * 1.025 * 0.9, dist ),
				);
				const rim = smoothstep( ghost.r * 0.55, ghost.r, dist ).mul( 0.7 ).add( 0.3 );
				const energy = ghost.k * 0.0028 / ( ghost.r * ghost.r );

				ghosts = ghosts.add( body.mul( rim ).mul( vec3( ghost.tint[ 0 ], ghost.tint[ 1 ], ghost.tint[ 2 ] ) ).mul( energy ) );

			}

			const offAxis = smoothstep( 0.02, 0.2, length( s ) );
			const centerDistance = length( p );
			const halo = vec3(
				smoothstep( 0.03, 0, abs( centerDistance.sub( 0.43 ) ) ),
				smoothstep( 0.03, 0, abs( centerDistance.sub( 0.445 ) ) ),
				smoothstep( 0.03, 0, abs( centerDistance.sub( 0.46 ) ) ),
			).mul( smoothstep( 0.25, 0.8, length( s ) ) ).mul( 0.12 );

			const q = p.sub( s );
			const rq = max( length( q ), 1e-4 );
			const aq = atan( q.y, q.x ).add( ROT );
			const px = float( 1 ).div( max( outputHeight, 1 ) );
			const spikeIndex = floor( aq.div( SPIKE ).add( 0.5 ) );
			const perp = abs( sin( aq.sub( spikeIndex.mul( SPIKE ) ) ) ).mul( rq );
			const wrapped = spikeIndex.sub( float( 2 * BLADES ).mul( floor( spikeIndex.div( 2 * BLADES ) ) ) );
			const h1 = fract( sin( wrapped.mul( 12.9898 ).add( 4.1 ) ).mul( 43758.5453 ) );
			const h2 = fract( sin( wrapped.mul( 78.233 ).add( 1.7 ) ).mul( 43758.5453 ) );
			const width = px.mul( 0.8 ).add( rq.mul( 0.003 ) );
			const line = exp( perp.mul( perp ).div( width.mul( width ) ).mul( - 1 ) ).mul( px.mul( 0.8 ).div( width ) );
			const reach = float( 0.05 ).add( h1.mul( 0.08 ) );
			const reachRgb = vec3( reach.mul( 1.12 ), reach, reach.mul( 0.88 ) );
			const taper = exp( rq.div( reachRgb ).mul( - 1 ).sub( rq.mul( rq ).div( reachRgb.mul( reachRgb ).mul( 4 ) ) ) );
			const spikes = line.mul( taper ).mul( float( 0.45 ).add( h2.mul( 0.55 ) ) ).mul( 1.4 );

			const bin = aq.mul( 1 / ( 2 * Math.PI ) ).add( 0.5 ).mul( 48 );
			const binIndex = floor( bin );
			const noise = mix( hashBin( wrapBin( binIndex ) ), hashBin( wrapBin( binIndex.add( 1 ) ) ), smoothstep( 0, 1, bin.sub( binIndex ) ) );
			const fine = pow( noise, 5 ).mul( exp( rq.div( 0.03 ).mul( - 1 ) ) ).mul( 0.3 );
			const glow = exp( rq.mul( - 5 ) ).mul( 0.05 ).add( exp( rq.mul( - 40 ) ).mul( 0.4 ) );
			const burst = spikes.add( fine ).add( glow );
			const lit = light.mul( ghosts.mul( offAxis ).add( halo ).add( burst ) );

			return vis.greaterThan( 0 ).select( lit, vec3( 0, 0, 0 ) );

		} )() as Node< 'vec3' >;

	}

	/**
	 * 太阳方向上云的透射。厚云里剩下的那一点透射按 `0.004..0.04` 的门槛收成 0。
	 *
	 * @param sunDirection 真太阳方向
	 */
	private cloudTransmittance( sunDirection: Node< 'vec3' > ): Node< 'float' > {

		if ( ! this.clouds ) return float( 1 );

		const transmitted = sampleSkyPro( this.clouds.viewClouds, this.clouds.viewCamera, sunDirection ).a;

		return transmitted.mul( smoothstep( 0.004, 0.04, transmitted ) );

	}

}

/** 七边形光圈的半径。边落在传入的距离比例上。 */
function apertureDistance( offset: Node< 'vec2' > ): Node< 'float' > {

	const angle = atan( offset.y, offset.x ).add( ROT );

	return length( offset ).mul( cos( floor( angle.div( SEG ).add( 0.5 ) ).mul( SEG ).sub( angle ) ) );

}

function wrapBin( index: Node< 'float' > ): Node< 'float' > {

	return index.sub( float( 48 ).mul( floor( index.div( 48 ) ) ) );

}

function hashBin( index: Node< 'float' > ): Node< 'float' > {

	return fract( sin( index.mul( 91.345 ) ).mul( 43758.5453 ) );

}
