/**
 * 调色。跟在自动曝光后面，还在线性 HDR 里。
 *
 * 这几项运行时改 `.value`。
 * 暖色 `0.02`、饱和度 `1.06`、对比度 `1.04`、暗角 `0.28`、颗粒 `0.012`。
 * ACES、sRGB、抖动在 `display.ts`。
 */

import { type Node } from 'three/webgpu';
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

/** 调色项。 */
export const gradeParams = {
	warmth: uniform( 0.02 ),
	saturation: uniform( 1.06 ),
	contrast: uniform( 1.04 ),
	vignette: uniform( 0.28 ),
	grain: uniform( 0.012 ),
};

const frameIndex = uniform( 0, 'uint' );

/** 每帧加一，颗粒才不会停在原地。 */
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

/** 暖色、饱和、对比、暗角、颗粒。 */
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
		const noise = postHash( pixel, frameIndex ).add( postHash( pixel.add( uvec2( uint( 7919 ), uint( 104729 ) ) ), frameIndex ) ).sub( 1 ).mul( 0.5 );

		return vec4( shaded.add( shaded.mul( noise.mul( gradeParams.grain ) ) ), color.a );

	} )() as Node< 'vec4' >;

}
