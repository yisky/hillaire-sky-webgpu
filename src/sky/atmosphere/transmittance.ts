/**
 * 透射查找表。
 *
 * 每个纹素是从大气内一点沿一个天顶角走到大气顶的透射 `exp(-光学厚度)`。
 * 参数化是 Bruneton 的 `(ρ, d)` 的逆映射，Hillaire EGSR 2020 第 5.2 节。
 * 积分与 Tidewater `Atmosphere.js` 的透射 kernel 相同：40 个均匀步，步长中点取样。
 * 横轴越靠左越接近天顶，纵轴从地面到大气顶。
 */

import {
	ClampToEdgeWrapping,
	HalfFloatType,
	LinearFilter,
	StorageTexture,
	type ComputeNode,
	type Node,
} from 'three/webgpu';
import { float, Fn, Loop, max, sqrt, textureStore, uniform, vec2, vec3, vec4, globalId } from 'three/tsl';

import { ATMOSPHERE_RADIUS_KM, GROUND_RADIUS_KM, TRANSMITTANCE_HEIGHT, TRANSMITTANCE_WIDTH } from '../constants';
import { atmosphereMedium, atmosphereRaySphereNearest } from './medium';

/** 沿大气顶的均匀样本数。 */
const TRANSMITTANCE_STEPS = 40;

/**
 * 为本实例创建透射表和写入它的 compute。
 *
 * 表是 `256×64` 的 `rgba16float`。三个缩放是这个实例自己的 uniform，
 * 不和其他天空共用。地面反照率不进入这张表。
 *
 * @returns 纹理、compute，以及计算时读取的三个缩放
 */
export function createTransmittanceLut() {

	const rayleighScale = uniform( 1 );
	const mieScale = uniform( 1 );
	const ozoneScale = uniform( 1 );

	const texture = new StorageTexture( TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT );
	texture.name = 'atmoTransmittance';
	texture.type = HalfFloatType;
	texture.magFilter = LinearFilter;
	texture.minFilter = LinearFilter;
	texture.wrapS = ClampToEdgeWrapping;
	texture.wrapT = ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.flipY = false;
	( texture as StorageTexture & { mipmapsAutoUpdate: boolean } ).mipmapsAutoUpdate = false;

	const compute = Fn( () => {

		const texel = vec2( float( globalId.x ), float( globalId.y ) ).add( 0.5 ).div( vec2( TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT ) ).toVar();
		// 写入时把纹素中心映回单位参数。采样时的内缩是它的逆。
		const xMu = texel.x.sub( 0.5 / TRANSMITTANCE_WIDTH ).mul( TRANSMITTANCE_WIDTH / ( TRANSMITTANCE_WIDTH - 1 ) );
		const xR = texel.y.sub( 0.5 / TRANSMITTANCE_HEIGHT ).mul( TRANSMITTANCE_HEIGHT / ( TRANSMITTANCE_HEIGHT - 1 ) );
		const H = sqrt( float( ATMOSPHERE_RADIUS_KM * ATMOSPHERE_RADIUS_KM - GROUND_RADIUS_KM * GROUND_RADIUS_KM ) );
		const rho = xR.mul( H );
		const radius = sqrt( rho.mul( rho ).add( GROUND_RADIUS_KM * GROUND_RADIUS_KM ) ).toVar();
		const dMin = float( ATMOSPHERE_RADIUS_KM ).sub( radius );
		const dMax = rho.add( H );
		const distance = dMin.add( xMu.mul( dMax.sub( dMin ) ) ).toVar();
		// d = 0 时方向就是天顶，余弦为 1。其余由弦长反解 μ。
		const mu = distance.equal( 0 ).select(
			1,
			H.mul( H ).sub( rho.mul( rho ) ).sub( distance.mul( distance ) ).div( radius.mul( distance ).mul( 2 ) ),
		).clamp( - 1, 1 ).toVar();

		const origin = vec3( 0, radius, 0 );
		const direction = vec3( sqrt( max( float( 1 ).sub( mu.mul( mu ) ), 0 ) ), mu, 0 );
		const limit = atmosphereRaySphereNearest( origin, direction, ATMOSPHERE_RADIUS_KM ).toVar();
		const step = limit.div( TRANSMITTANCE_STEPS );
		const opticalDepth = vec3( 0 ).toVar();

		Loop( TRANSMITTANCE_STEPS, ( { i } ) => {

			const traveled = float( i ).add( 0.5 ).mul( step );
			const position = origin.add( direction.mul( traveled ) );
			const altitude = position.length().sub( GROUND_RADIUS_KM );
			const extinction = atmosphereMedium( altitude, rayleighScale, mieScale, ozoneScale ).get( 'extinction' ) as Node< 'vec3' >;

			opticalDepth.addAssign( extinction.mul( step ) );

		} );

		textureStore( texture, globalId.xy, vec4( opticalDepth.negate().exp(), 1 ) ).toWriteOnly();

	} )().computeKernel( [ 8, 8, 1 ] );

	compute.dispatchSize = [ TRANSMITTANCE_WIDTH / 8, TRANSMITTANCE_HEIGHT / 8, 1 ];
	compute.setName( 'Atmosphere.Transmittance' );

	return { texture, compute: compute as ComputeNode, rayleighScale, mieScale, ozoneScale };

}
