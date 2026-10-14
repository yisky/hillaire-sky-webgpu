/**
 * 显示变换：ACES、sRGB，再加 ±1/255 抖动打散色带。
 * 接在调色后面。曝光默认 `0.55`，界面按 EV 写成 `0.55 * 2^ev`。
 */

import { type Node } from 'three/webgpu';
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

/** 色调映射前的曝光倍数。 */
export const exposure = uniform( 0.55 );

const frameIndex = uniform( 0, 'uint' );

/** 每帧加一，抖动才不会停在原地。 */
export function updateDisplay(): void {

	frameIndex.value = ( Number( frameIndex.value ) + 1 ) >>> 0;

}

/** 像素哈希，结果在 0..1。两路相减得到约 ±1 的抖动。 */
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

/** 线性 HDR → ACES → sRGB → 抖动。 */
export function display( color: Node< 'vec4' > ): Node< 'vec4' > {

	return Fn( () => {

		const mapped = acesFilmicToneMapping( color.rgb, exposure ) as Node< 'vec3' >;
		const pixel = uvec2( viewportCoordinate );
		const frame = frameIndex;
		const dither = postHash( pixel.add( uvec2( uint( 31337 ), uint( 271 ) ) ), frame )
			.add( postHash( pixel.add( uvec2( uint( 1013 ), uint( 65537 ) ) ), frame ) )
			.sub( 1 )
			.div( 255 );

		return vec4( linearToSrgb( mapped ).add( dither ), 1 );

	} )() as Node< 'vec4' >;

}
