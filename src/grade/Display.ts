/**
 * 显示变换。接在调色后面，管线最后一步。
 *
 * 线性 HDR 先乘曝光、走 ACES 压进大约 0..1，再按 sRGB 编码，最后加 ±1/255 抖动打色散带。
 * 曝光默认 `0.55`，界面按 EV 写成 `0.55 * 2^ev`。ACES、sRGB、抖动必须连在一起，中间不插别的节点。
 *
 * 接法跟调色一样走 {@link TempNode}：图里直接用这个节点，`setup` 返回显示器该看到的颜色。
 * `updateBefore` 只写帧序号，抖动才不会停在原地。
 * 管线要关掉自带的输出变换，避免和这里的 ACES、sRGB 叠两层。
 */

import { NodeUpdateType, TempNode, type Node, type NodeFrame } from 'three/webgpu';
import {
	Fn,
	acesFilmicToneMapping,
	float,
	uint,
	uniform,
	uvec2,
	vec3,
	vec4,
	viewportCoordinate,
} from 'three/tsl';

/**
 * 构造显示变换时的输入。曝光是实例上的 uniform，构造之后写 `.value`。
 */
export interface DisplayOptions {

	/** 已经调过色的颜色。还在线性 HDR 里。 */
	input: Node< 'vec4' >;

}

/**
 * 显示变换。
 *
 * 构造时传入已经调过色的颜色。输出节点就是这个节点。
 * 每帧抄好 {@link Display.exposure}，再 `pipeline.render()`。没有单独的 compute。
 */
export class Display extends TempNode {

	/** 色调映射前的曝光倍数。默认 `0.55`。 */
	readonly exposure = uniform( 0.55 );

	/** 帧序号。抖动哈希用它，每帧加一。 */
	private readonly frameIndex = uniform( 0, 'uint' );

	/** 已经调过色的颜色。 */
	private readonly input: Node< 'vec4' >;

	/**
	 * @param options 要交出去显示的颜色
	 */
	constructor( options: DisplayOptions ) {

		super( 'vec4' );

		this.updateBeforeType = NodeUpdateType.FRAME;
		this.input = options.input;

	}

	/**
	 * ACES、sRGB、抖动。接在调色后面，作为管线输出。
	 */
	setup(): Node {

		const color = this.input;
		const exposure = this.exposure;
		const frameIndex = this.frameIndex;

		return Fn( () => {

			const mapped = acesFilmicToneMapping( color.rgb, exposure ) as Node< 'vec3' >;
			const pixel = uvec2( viewportCoordinate );
			const dither = postHash( pixel.add( uvec2( uint( 31337 ), uint( 271 ) ) ), frameIndex )
				.add( postHash( pixel.add( uvec2( uint( 1013 ), uint( 65537 ) ) ), frameIndex ) )
				.sub( 1 )
				.div( 255 );

			return vec4( linearToSrgb( mapped ).add( dither ), 1 );

		} )() as Node< 'vec4' >;

	}

	/**
	 * 帧序号加一。抖动哈希用它，色调映射本身不用。
	 *
	 * @param _frame 当前节点帧
	 */
	updateBefore( _frame: NodeFrame ): boolean | undefined {

		this.frameIndex.value = ( Number( this.frameIndex.value ) + 1 ) >>> 0;

		return;

	}

}

/** 像素哈希，结果在 0..1。两路相减再除 255，得到约 ±1/255 的抖动。 */
function postHash( pixel: Node< 'uvec2' >, frame: Node< 'uint' > ): Node< 'float' > {

	const x = pixel.x.mul( uint( 1664525 ) ).add( pixel.y.mul( uint( 1013904223 ) ) ).add( frame.mul( uint( 2654435761 ) ) ).toVar();

	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) );
	x.assign( x.mul( uint( 0x7feb352d ) ) );
	x.assign( x.bitXor( x.shiftRight( uint( 15 ) ) ) );
	x.assign( x.mul( uint( 0x846ca68b ) ) );
	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) );

	return float( x.shiftRight( uint( 8 ) ) ).div( 16777216 );

}

/** 线性到 sRGB。小值走 12.92 那段，避免暗部被幂次抬起来。 */
function linearToSrgb( color: Node< 'vec3' > ): Node< 'vec3' > {

	const encoded = color.pow( float( 1 ).div( 2.4 ) ).mul( 1.055 ).sub( 0.055 );
	const linear = color.mul( 12.92 );
	const cutoff = float( 0.0031308 );

	return vec3(
		color.x.lessThanEqual( cutoff ).select( linear.x, encoded.x ),
		color.y.lessThanEqual( cutoff ).select( linear.y, encoded.y ),
		color.z.lessThanEqual( cutoff ).select( linear.z, encoded.z ),
	);

}
