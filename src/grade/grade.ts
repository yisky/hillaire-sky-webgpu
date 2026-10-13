/**
 * 调色。跟在自动曝光后面、输出前面。
 *
 * 这几项对应 Tidewater 的 `PostParams`，运行时改 `.value`。
 * 暖色 `0.02`、饱和度 `1.06`、对比度 `1.04`、暗角 `0.28`、颗粒 `0.012`。
 * `exposure` 对应 `frame.exposure`，默认 `0.55`。Tidewater 的界面按 EV 写成 `0.55 * 2^ev`。
 * 后面是 ACES、sRGB，最后加 ±1/255 的抖动。
 */

import { type Node } from 'three/webgpu';
import {
	Fn,
	acesFilmicToneMapping,
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

/** Tidewater `PostParams` 里的调色项。 */
export const gradeParams = {
	warmth: uniform( 0.02 ),
	saturation: uniform( 1.06 ),
	contrast: uniform( 1.04 ),
	vignette: uniform( 0.28 ),
	grain: uniform( 0.012 ),
};

/** Tidewater 的 `frame.exposure`。 */
export const exposure = uniform( 0.55 );

const frameIndex = uniform( 0, 'uint' );

/** 每帧加一，颗粒和抖动才不会停在原地。 */
export function updateGrade(): void {

	frameIndex.value = ( Number( frameIndex.value ) + 1 ) >>> 0;

}

function postHash( pixel: Node< 'uvec2' >, frame: Node< 'uint' > ): Node< 'float' > {

	const x = pixel.x.mul( uint( 1664525 ) ).add( pixel.y.mul( uint( 1013904223 ) ) ).add( frame.mul( uint( 2654435761 ) ) ).toVar();

	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) );
	x.assign( x.mul( uint( 0x7feb352d ) ) );
	x.assign( x.bitXor( x.shiftRight( uint( 15 ) ) ) );
	x.assign( x.mul( uint( 0x846ca68b ) ) );
	x.assign( x.bitXor( x.shiftRight( uint( 16 ) ) ) );

	return float( x.shiftRight( uint( 8 ) ) ).div( 16777216 );

}

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

/** 暖色、饱和、对比、暗角、颗粒，然后 ACES、sRGB 和抖动。 */
export function grade( color: Node< 'vec4' > ): Node< 'vec4' > {

	return Fn( () => {

		const warmth = gradeParams.warmth;
		const warmed = color.rgb.mul( vec3( warmth.add( 1 ), 1, float( 1 ).sub( warmth ) ) );
		const gray = luminance( warmed );
		const saturated = mix( vec3( gray ), warmed, gradeParams.saturation );
		const contrasted = pow( max( saturated, vec3( 0, 0, 0 ) ).div( 0.18 ), vec3( gradeParams.contrast ) ).mul( 0.18 );
		const edge = screenUV.sub( 0.5 ).mul( vec2( 1, 0.8 ) );
		const vignette = float( 1 ).sub( smoothstep( 0.25, 0.75, length( edge ) ).mul( gradeParams.vignette ) );
		const shaded = contrasted.mul( vignette );
		const pixel = uvec2( viewportCoordinate );
		const frame = frameIndex;
		const noise = postHash( pixel, frame ).add( postHash( pixel.add( uvec2( uint( 7919 ), uint( 104729 ) ) ), frame ) ).sub( 1 ).mul( 0.5 );
		const graded = shaded.add( shaded.mul( noise.mul( gradeParams.grain ) ) );
		const mapped = acesFilmicToneMapping( graded, exposure ) as Node< 'vec3' >;
		const dither = postHash( pixel.add( uvec2( uint( 31337 ), uint( 271 ) ) ), frame )
			.add( postHash( pixel.add( uvec2( uint( 1013 ), uint( 65537 ) ) ), frame ) )
			.sub( 1 )
			.div( 255 );

		return vec4( linearToSrgb( mapped ).add( dither ), 1 );

	} )() as Node< 'vec4' >;

}
