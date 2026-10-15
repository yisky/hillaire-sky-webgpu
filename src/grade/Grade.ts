/**
 * 调色。跟在自动曝光后面，还在线性 HDR 里。
 *
 * 暖色、饱和、对比、暗角、颗粒。这几项运行时改 `.value`。
 * 默认暖色 `0.02`、饱和度 `1.06`、对比度 `1.04`、暗角 `0.28`、颗粒 `0.012`。
 * ACES、sRGB、抖动在显示变换里做。
 *
 * 接法跟泛光、炫光、雾、自动曝光一样走 {@link TempNode}：图里直接用这个节点，
 * `setup` 返回调过色的颜色。`updateBefore` 只写帧序号，颗粒才不会停在原地。
 */

import { NodeUpdateType, TempNode, type Node, type NodeFrame } from 'three/webgpu';
import {
	Fn,
	float,
	length,
	luminance,
	max,
	mix,
	pow,
	screenUV,
	smoothstep,
	uint,
	uniform,
	uvec2,
	vec2,
	vec3,
	vec4,
	viewportCoordinate,
} from 'three/tsl';

/**
 * 构造调色时的输入。暖色、饱和这些是实例上的 uniform，构造之后写 `.value`。
 */
export interface GradeOptions {

	/** 已经做过自动曝光的颜色。`setup` 在它上面调色。 */
	input: Node< 'vec4' >;

}

/**
 * 调色。
 *
 * 构造时传入已经做过自动曝光的颜色。显示变换的输入就是这个节点。
 * 每帧抄好饱和度、对比、暗角、颗粒，再 `pipeline.render()`。没有单独的 compute。
 */
export class Grade extends TempNode {

	/** 暖色。正值偏红减蓝。 */
	readonly warmth = uniform( 0.02 );

	/** 饱和度。`1` 是原样，小于 `1` 往灰靠。 */
	readonly saturation = uniform( 1.06 );

	/** 对比度。绕中灰 `0.18` 做幂次。 */
	readonly contrast = uniform( 1.04 );

	/** 暗角强度。`0` 关掉。 */
	readonly vignette = uniform( 0.28 );

	/** 颗粒强度。`0` 关掉。 */
	readonly grain = uniform( 0.012 );

	/** 帧序号。颗粒哈希用它，每帧加一。 */
	private readonly frameIndex = uniform( 0, 'uint' );

	/** 已经做过自动曝光的颜色。 */
	private readonly input: Node< 'vec4' >;

	/**
	 * @param options 要调色的颜色
	 */
	constructor( options: GradeOptions ) {

		super( 'vec4' );

		this.updateBeforeType = NodeUpdateType.FRAME;
		this.input = options.input;

	}

	/**
	 * 暖色、饱和、对比、暗角、颗粒。接在自动曝光后面、显示变换前面。
	 */
	setup(): Node {

		const color = this.input;
		const warmth = this.warmth;
		const saturation = this.saturation;
		const contrast = this.contrast;
		const vignetteAmount = this.vignette;
		const grain = this.grain;
		const frameIndex = this.frameIndex;

		return Fn( () => {

			const warmed = color.rgb.mul( vec3( warmth.add( 1 ), 1, float( 1 ).sub( warmth ) ) );
			const gray = luminance( warmed );
			const saturated = mix( vec3( gray ), warmed, saturation );
			// 绕中灰做幂次，暗部不会被对比度抬成死黑。
			const contrasted = pow( max( saturated, vec3( 0, 0, 0 ) ).div( 0.18 ), vec3( contrast ) ).mul( 0.18 );
			const edge = screenUV.sub( 0.5 ).mul( vec2( 1, 0.8 ) );
			const vignette = float( 1 ).sub( smoothstep( 0.25, 0.75, length( edge ) ).mul( vignetteAmount ) );
			const shaded = contrasted.mul( vignette );
			const pixel = uvec2( viewportCoordinate );
			const noise = postHash( pixel, frameIndex ).add( postHash( pixel.add( uvec2( uint( 7919 ), uint( 104729 ) ) ), frameIndex ) ).sub( 1 ).mul( 0.5 );

			return vec4( shaded.add( shaded.mul( noise.mul( grain ) ) ), color.a );

		} )() as Node< 'vec4' >;

	}

	/**
	 * 帧序号加一。颗粒哈希用它，调色本身不用。
	 *
	 * @param _frame 当前节点帧
	 */
	updateBefore( _frame: NodeFrame ): boolean | undefined {

		this.frameIndex.value = ( Number( this.frameIndex.value ) + 1 ) >>> 0;

		return;

	}

}

/** 像素哈希，结果在 0..1。两路相减得到约 ±1 的颗粒。 */
function postHash( pixel: Node< 'uvec2' >, frame: Node< 'uint' > ): Node< 'float' > {

	const x = pixel.x.mul( uint( 1664525 ) ).add( pixel.y.mul( uint( 1013904223 ) ) ).add( frame.mul( uint( 2654435761 ) ) ).toVar();

	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) );
	x.assign( x.mul( uint( 0x7feb352d ) ) );
	x.assign( x.bitXor( x.shiftRight( uint( 15 ) ) ) );
	x.assign( x.mul( uint( 0x846ca68b ) ) );
	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) );

	return float( x.shiftRight( uint( 8 ) ) ).div( 16777216 );

}
