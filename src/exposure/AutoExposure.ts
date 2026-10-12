/**
 * 自动曝光。单独的后期组件，不读天空模块。
 *
 * 整幅画面乘同一个倍数，把中心加权的对数平均亮度慢慢拉向 `0.25`。
 * 偏暗时加得快（每秒 `1.6`），偏亮时减得慢（每秒 `1.1`）。倍数限制在 `0.6` 到 `6`。
 * 太阳低于地平线时上限收到 `2`，夜空不会被拉成白天。阈值和天空里恒星出现的区间一致：
 * `-sun.y` 从 `0.02` 到 `0.18`。
 * 倍数晚一帧生效。源码量的是泛光缩到 `1/16` 的那张图；这里还没有泛光，改成在场景颜色上取 `32×32` 个点。
 */

import {
	StorageBufferAttribute,
	Vector3,
	type ComputeNode,
	type Node,
	type Texture,
} from 'three/webgpu';
import {
	Fn,
	Loop,
	exp,
	exp2,
	float,
	floor,
	int,
	ivec2,
	length,
	log2,
	luminance,
	max,
	mix,
	pow,
	smoothstep,
	storage,
	texture,
	textureLoad,
	textureSize,
	vec2,
	vec4,
	uniform,
} from 'three/tsl';

const GRID = 32;
const SAMPLES = GRID * GRID;

const REF = 0.25;
const MIN = 0.6;
const MAX = 6;
const UP = 1.6;
const DOWN = 1.1;
const NIGHT_MAX = 2;

export interface AutoExposureOptions {

	/** 场景通道的颜色。计量读它，不读已经加过炫光的结果。 */
	color: Texture;

	/** 指向太阳的世界方向。调用方原地修改即可。 */
	sunDirection: Vector3;

}

/**
 * 自动曝光。
 *
 * 每帧先 {@link AutoExposure.update}，再画后期，最后 `renderer.compute(exposure.adapt)`。
 */
export class AutoExposure {

	/** 适应。场景画完后派发，下一帧才用新的倍数。 */
	readonly adapt: ComputeNode;

	private readonly sunDirection: Vector3;

	private readonly dt = uniform( 1 / 60 );

	private readonly factor: Node< 'float' >;

	/**
	 * @param options 场景颜色和太阳方向
	 */
	constructor( options: AutoExposureOptions ) {

		this.sunDirection = options.sunDirection;

		const attribute = new StorageBufferAttribute( new Float32Array( [ 1 ] ), 1 );
		const written = storage( attribute, 'float', 1 );
		const read = storage( attribute, 'float', 1 ).toReadOnly();
		const sunDirection = uniform( options.sunDirection ) as Node< 'vec3' >;

		this.factor = read.element( 0 ) as Node< 'float' >;
		this.adapt = this.createAdapt( written.element( 0 ) as Node< 'float' >, options.color, sunDirection );

	}

	/**
	 * 用当前倍数缩放颜色。
	 *
	 * @param color 已经叠过炫光的场景颜色
	 */
	apply( color: Node< 'vec4' > ): Node< 'vec4' > {

		return vec4( color.rgb.mul( this.factor ), color.a );

	}

	/**
	 * 写下这一帧的时间间隔。在画后期之前调用。
	 *
	 * @param delta 距上一帧的秒数
	 */
	update( delta: number ): void {

		this.dt.value = Math.min( Math.max( delta, 0 ), 0.1 );

	}

	/**
	 * 量这一帧的亮度，并写上下一帧要用的倍数。
	 *
	 * @param current 可写的倍数，初值是 `1`
	 * @param color 场景颜色
	 * @param sunDirection 指向太阳的方向
	 */
	private createAdapt( current: Node< 'float' >, color: Texture, sunDirection: Node< 'vec3' > ): ComputeNode {

		const image = texture( color );
		const dt = this.dt;

		return Fn( () => {

			const size = textureSize( image, int( 0 ) ) as unknown as Node< 'vec2' >;
			const accL = float( 0 ).toVar();
			const accW = float( 0 ).toVar();

			Loop( SAMPLES, ( input ) => {

				const index = float( ( input as unknown as { i: Node< 'int' > } ).i );
				const column = index.mod( GRID );
				const row = floor( index.div( GRID ) );
				const uv = vec2( column.add( 0.5 ), row.add( 0.5 ) ).div( GRID );
				const coord = ivec2( uv.mul( size ) );
				const sample = textureLoad( image, coord, int( 0 ) ) as unknown as Node< 'vec4' >;
				const centered = uv.sub( 0.5 );
				const weight = max( float( 1 ).sub( length( centered.mul( vec2( 1, 1.4 ) ) ).mul( 1.2 ) ), 0.15 );
				const logLum = log2( max( luminance( sample.rgb ), 1e-4 ) );

				accL.addAssign( logLum.mul( weight ) );
				accW.addAssign( weight );

			} );

			const average = exp2( accL.div( max( accW, 1e-4 ) ) );
			const ratio = float( REF ).div( average );
			const partial = ratio.greaterThan( 1 ).select( pow( ratio, 0.8 ), ratio );
			const night = smoothstep( 0.02, 0.18, sunDirection.y.negate() );
			const cap = mix( MAX, NIGHT_MAX, night );
			const target = partial.clamp( MIN, cap );
			const rate = target.greaterThan( current ).select( UP, DOWN );
			const step = float( 1 ).sub( exp( dt.mul( rate ).negate() ) );
			const next = exp2( mix( log2( max( current, 1e-3 ) ), log2( target ), step ) );

			current.assign( next );

		} )().compute( 1 ) as ComputeNode;

	}

}
