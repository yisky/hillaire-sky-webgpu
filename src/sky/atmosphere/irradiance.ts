/**
 * 天空辐照度积分。
 *
 * 对天空视图表做余弦加权半球积分，Hillaire 把结果存成 `E/π`，也就是这些样本的平均亮度。
 * 另外两个 `vec4` 是海平面沿太阳方向的透射，以及地平线一圈的平均颜色。
 * 单线程、`16×16` 个半球样本、地平线 16 个方位。距离单位为千米。
 */

import {
	StorageBufferAttribute,
	type ComputeNode,
	type Node,
	type StorageTexture,
} from 'three/webgpu';
import {
	Fn,
	Loop,
	cos,
	float,
	max,
	normalize,
	sin,
	sqrt,
	storage,
	vec3,
	vec4,
} from 'three/tsl';

import { GROUND_RADIUS_KM } from '../constants';
import { sampleTransmittance } from './multiScattering';
import { sampleSkyView } from './skyView';

/** 余弦加权半球每边的样本数。总样本 `16×16`。 */
const HEMISPHERE_RESOLUTION = 16;

/** 地平线一圈的方位样本数。 */
const HORIZON_SAMPLES = 16;

/**
 * 创建辐照积分和它写入的缓冲。
 *
 * 缓冲是 3 个 `vec4`：`[0]` 为 `E/π`，`[1]` 为海平面太阳透射，`[2]` 为地平线颜色。
 * 每项的 `w` 写 1。必须在天空视图表更新之后派发，积分读的是这张表。
 *
 * @param skyView 天空视图表
 * @param transmittance 透射表
 * @param viewHeight 与天空视图相同的地心距离，单位 km
 * @param sunDirection 世界空间太阳方向
 * @returns 存储缓冲和 compute
 */
export function createIrradiance(
	skyView: StorageTexture,
	transmittance: StorageTexture,
	viewHeight: Node< 'float' >,
	sunDirection: Node,
) {

	const attribute = new StorageBufferAttribute( 3, 4 );
	const buffer = storage( attribute, 'vec4', 3 );
	const hemisphereLoop = { start: 0, end: HEMISPHERE_RESOLUTION * HEMISPHERE_RESOLUTION, type: 'int' as const, condition: '<', name: 'sample' };
	const horizonLoop = { start: 0, end: HORIZON_SAMPLES, type: 'int' as const, condition: '<', name: 'horizon' };

	const compute = Fn( () => {

		const sun = sunDirection as Node< 'vec3' >;
		const luminance = vec3( 0 ).toVar();

		Loop( hemisphereLoop, ( input ) => {

			const index = float( ( input as unknown as { sample: Node< 'int' > } ).sample );
			const horizontal = index.mod( HEMISPHERE_RESOLUTION ).add( 0.5 ).div( HEMISPHERE_RESOLUTION );
			const vertical = index.div( HEMISPHERE_RESOLUTION ).floor().add( 0.5 ).div( HEMISPHERE_RESOLUTION );
			const radius = sqrt( vertical );
			const phi = horizontal.mul( Math.PI * 2 );
			const direction = vec3( radius.mul( cos( phi ) ), sqrt( max( float( 1 ).sub( vertical ), 0 ) ), radius.mul( sin( phi ) ) );

			luminance.addAssign( sampleSkyView( skyView, viewHeight, sun, direction ) );

		} );

		buffer.element( 0 ).assign( vec4( luminance.div( HEMISPHERE_RESOLUTION * HEMISPHERE_RESOLUTION ), 1 ) );

		const transmitted = sampleTransmittance( transmittance, float( GROUND_RADIUS_KM + 0.001 ), sun.y );

		buffer.element( 1 ).assign( vec4( transmitted, 1 ) );

		const horizon = vec3( 0 ).toVar();

		Loop( horizonLoop, ( input ) => {

			const index = float( ( input as unknown as { horizon: Node< 'int' > } ).horizon );
			const phi = index.mul( Math.PI * 2 / HORIZON_SAMPLES );
			const direction = normalize( vec3( cos( phi ), 0.03, sin( phi ) ) );

			horizon.addAssign( sampleSkyView( skyView, viewHeight, sun, direction ) );

		} );

		buffer.element( 2 ).assign( vec4( horizon.div( HORIZON_SAMPLES ), 1 ) );

	} )().compute( 1 );

	compute.setName( 'Atmosphere.Irradiance' );

	return { attribute, compute: compute as ComputeNode };

}
