/**
 * 自动曝光。单独的后期组件，不读天空模块。
 *
 * 整幅画面乘同一个倍数，把中心加权的对数平均亮度慢慢拉向 `ref`。
 * 偏暗时按 `up` 加快，偏亮时按 `down` 减慢。倍数限制在 `min` 到 `max`。
 * `enabled` 小于等于 `0.5` 时倍数保持 `1`。这几项运行时改 `.value`。
 * 太阳低于地平线时上限收到 `2`，夜空不会被拉成白天。阈值和天空里恒星出现的区间一致：
 * `-sun.y` 从 `0.02` 到 `0.18`。
 * 倍数晚一帧生效。计量读泛光缩到 `1/16` 的那张图，一个 256 线程的组把每个纹素都加一遍。
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
	If,
	Loop,
	exp,
	exp2,
	float,
	int,
	ivec2,
	length,
	localId,
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
	uint,
	vec2,
	vec4,
	uniform,
	workgroupArray,
	workgroupBarrier,
} from 'three/tsl';

const GROUP = 256;
const SPAN = 160;
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

	/** 大于 `0.5` 才适应，否则倍数保持 `1`。Tidewater `autoExposure.enabled`。 */
	readonly enabled = uniform( 1 );

	/** 不需要修正的平均亮度。Tidewater 着色器里叫 `refLum`，因为 `ref` 是 WGSL 关键字。 */
	readonly ref = uniform( 0.25 );

	/** 倍数下限。Tidewater `autoExposure.min`。 */
	readonly min = uniform( 0.6 );

	/** 倍数上限。Tidewater `autoExposure.max`。夜里另收到 `2`。 */
	readonly max = uniform( 6 );

	/** 变亮的速度，单位 1/秒。Tidewater `autoExposure.up`。 */
	readonly up = uniform( 1.6 );

	/** 变暗的速度，单位 1/秒。Tidewater `autoExposure.down`。 */
	readonly down = uniform( 1.1 );

	private readonly dt = uniform( 1 / 60 );

	private readonly factor: Node< 'float' >;

	/**
	 * @param options 场景颜色和太阳方向
	 */
	constructor( options: AutoExposureOptions ) {

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
		const enabled = this.enabled;
		const ref = this.ref;
		const min = this.min;
		const exposureMax = this.max;
		const up = this.up;
		const down = this.down;
		const sumL = workgroupArray( 'float', GROUP ) as unknown as { element( index: Node ): Node< 'float' > };
		const sumW = workgroupArray( 'float', GROUP ) as unknown as { element( index: Node ): Node< 'float' > };

		const built = Fn( () => {

			const thread = localId.x;
			const size = textureSize( image, int( 0 ) ) as unknown as Node< 'uvec2' >;
			const count = size.x.mul( size.y );
			const accL = float( 0 ).toVar();
			const accW = float( 0 ).toVar();

			Loop( { start: uint( 0 ), end: uint( SPAN ), type: 'uint' }, ( { i } ) => {

				const index = thread.add( i.mul( uint( GROUP ) ) );

				If( index.lessThan( count ), () => {

					const x = index.mod( size.x );
					const y = index.div( size.x );
					const sample = textureLoad( image, ivec2( int( x ), int( y ) ), int( 0 ) ) as unknown as Node< 'vec4' >;
					const centered = vec2( float( x ).add( 0.5 ).div( float( size.x ) ), float( y ).add( 0.5 ).div( float( size.y ) ) ).sub( 0.5 );
					const weight = max( float( 1 ).sub( length( centered.mul( vec2( 1, 1.4 ) ) ).mul( 1.2 ) ), 0.15 );
					const logLum = log2( max( luminance( sample.rgb ), 1e-4 ) );

					accL.addAssign( logLum.mul( weight ) );
					accW.addAssign( weight );

				} );

			} );

			sumL.element( thread ).assign( accL );
			sumW.element( thread ).assign( accW );
			workgroupBarrier();

			for ( let span = GROUP / 2; span > 0; span >>= 1 ) {

				const step = uint( span );

				If( thread.lessThan( step ), () => {

					sumL.element( thread ).addAssign( sumL.element( thread.add( step ) ) );
					sumW.element( thread ).addAssign( sumW.element( thread.add( step ) ) );

				} );

				workgroupBarrier();

			}

			If( thread.equal( uint( 0 ) ), () => {

				const average = exp2( sumL.element( uint( 0 ) ).div( max( sumW.element( uint( 0 ) ), 1e-4 ) ) );
				const ratio = ref.div( average );
				const partial = ratio.greaterThan( 1 ).select( pow( ratio, 0.8 ), ratio );
				const night = smoothstep( 0.02, 0.18, sunDirection.y.negate() );
				const cap = mix( exposureMax, NIGHT_MAX, night );
				const target = partial.clamp( min, cap );
				const rate = target.greaterThan( current ).select( up, down );
				const step = float( 1 ).sub( exp( dt.mul( rate ).negate() ) );
				const next = exp2( mix( log2( max( current, 1e-3 ) ), log2( target ), step ) );

				current.assign( enabled.greaterThan( 0.5 ).select( next, float( 1 ) ) );

			} );

		} )();

		return ( built as unknown as { compute( count: number[], size: number[] ): ComputeNode } ).compute( [ 1, 1, 1 ], [ GROUP, 1, 1 ] );

	}

}
