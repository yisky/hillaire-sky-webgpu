/**
 * 多次散射查找表。
 *
 * 每个纹素是二阶散射经过各向同性相位闭合后的结果 `L / (1 - fms)`，
 * 对应 Hillaire EGSR 2020 式 (5)–(10)。
 * 横轴是太阳天顶角余弦，`-1` 在左侧、`1` 在右侧。纵轴是归一化高度，下为地面。
 * 多次散射积分按 `multiScatKernel` 的步数取样。
 * 距离单位为千米。表内的值还没乘太阳照度。
 */

import {
	ClampToEdgeWrapping,
	Color,
	HalfFloatType,
	LinearFilter,
	StorageTexture,
	type ComputeNode,
	type Node,
} from 'three/webgpu';
import {
	Fn,
	If,
	Loop,
	acos,
	cos,
	float,
	globalId,
	max,
	normalize,
	sin,
	sqrt,
	textureLevel,
	textureStore,
	uniform,
	vec2,
	vec3,
	vec4,
} from 'three/tsl';

import { ATMOSPHERE_RADIUS_KM, GROUND_RADIUS_KM, MULTI_SCATTER_SIZE } from '../constants';
import { atmosphereMedium, atmosphereRaySphereNearest, atmosphereTransmittanceUV } from './medium';

/** 方向网格边长。`8×8` 个方向均匀铺满球面。 */
const DIRECTION_GRID = 8;

/** 方向数。 */
const DIRECTIONS = DIRECTION_GRID * DIRECTION_GRID;

/** 每条光线的步数。 */
const STEPS = 20;

/** 各向同性相位 `1 / 4π`。 */
const ISOTROPIC_PHASE = 1 / ( 4 * Math.PI );

/** 球面均匀采样的立体角权重 `4π / N`。 */
const SPHERE_WEIGHT = ( 4 * Math.PI ) / DIRECTIONS;

/**
 * 为本实例创建多次散射表。
 *
 * 表是 `32×32` 的 `rgba16float`。散射缩放与透射表共用同一组 uniform。
 * 地面反照率是调用方持有的 `Color`，兰伯特反射读它。
 *
 * @param transmittance 刚写完的透射表。本 compute 只采样，不写入
 * @param rayleighScale 瑞利倍数
 * @param mieScale 米氏倍数
 * @param ozoneScale 臭氧倍数
 * @param groundAlbedo 地面反照率，线性 RGB，与宿主同一个 `Color`
 * @returns 纹理和 compute
 */
export function createMultiScatteringLut(
	transmittance: StorageTexture,
	rayleighScale: Node< 'float' >,
	mieScale: Node< 'float' >,
	ozoneScale: Node< 'float' >,
	groundAlbedo: Color,
) {

	const albedo = uniform( groundAlbedo );
	const texture = new StorageTexture( MULTI_SCATTER_SIZE, MULTI_SCATTER_SIZE );
	texture.name = 'atmoMultiScat';
	texture.type = HalfFloatType;
	texture.magFilter = LinearFilter;
	texture.minFilter = LinearFilter;
	texture.wrapS = ClampToEdgeWrapping;
	texture.wrapT = ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.flipY = false;
	( texture as StorageTexture & { mipmapsAutoUpdate: boolean } ).mipmapsAutoUpdate = false;

	const directionLoop = { start: 0, end: DIRECTIONS, type: 'int' as const, condition: '<', name: 'direction' };
	const stepLoop = { start: 0, end: STEPS, type: 'int' as const, condition: '<', name: 'step' };

	const compute = Fn( () => {

		const texel = vec2( float( globalId.x ), float( globalId.y ) ).add( 0.5 ).div( MULTI_SCATTER_SIZE ).toVar();
		const uv = texel.sub( 0.5 / MULTI_SCATTER_SIZE ).mul( MULTI_SCATTER_SIZE / ( MULTI_SCATTER_SIZE - 1 ) );
		const cosSun = uv.x.mul( 2 ).sub( 1 ).toVar();
		const radius = float( GROUND_RADIUS_KM ).add( uv.y.clamp( 0.001, 0.999 ).mul( ATMOSPHERE_RADIUS_KM - GROUND_RADIUS_KM ) ).toVar();
		const sunDirection = normalize( vec3( 0, cosSun, sqrt( max( float( 1 ).sub( cosSun.mul( cosSun ) ), 0 ) ).negate() ) );
		const origin = vec3( 0, radius, 0 );
		const luminanceSum = vec3( 0 ).toVar();
		const gatheredSum = vec3( 0 ).toVar();

		Loop( directionLoop, ( input ) => {

			const directionIndex = float( ( input as unknown as { direction: Node< 'int' > } ).direction );
			const column = directionIndex.mod( DIRECTION_GRID ).add( 0.5 ).div( DIRECTION_GRID );
			const row = directionIndex.div( DIRECTION_GRID ).floor().add( 0.5 ).div( DIRECTION_GRID );
			const theta = column.mul( Math.PI * 2 );
			const phi = acos( float( 1 ).sub( row.mul( 2 ) ) );
			const sinPhi = sin( phi );
			const rayDirection = vec3( cos( theta ).mul( sinPhi ), cos( phi ), sin( theta ).mul( sinPhi ) );

			const tGround = atmosphereRaySphereNearest( origin, rayDirection, GROUND_RADIUS_KM ).toVar();
			const tTop = atmosphereRaySphereNearest( origin, rayDirection, ATMOSPHERE_RADIUS_KM ).toVar();
			const hitGround = tGround.greaterThan( 0 );
			const limit = hitGround.select( tGround, tTop ).toVar();
			const stepSize = limit.div( STEPS );
			const throughput = vec3( 1 ).toVar();
			const luminance = vec3( 0 ).toVar();
			const gathered = vec3( 0 ).toVar();

			Loop( stepLoop, ( stepInput ) => {

				const traveled = float( ( stepInput as unknown as { step: Node< 'int' > } ).step ).add( 0.3 ).mul( stepSize );
				const position = origin.add( rayDirection.mul( traveled ) );
				const distance = position.length().toVar();
				const up = position.div( distance );
				const medium = atmosphereMedium( distance.sub( GROUND_RADIUS_KM ), rayleighScale, mieScale, ozoneScale );
				const scattering = medium.get( 'scattering' ) as Node< 'vec3' >;
				const extinction = medium.get( 'extinction' ) as Node< 'vec3' >;
				const sunTransmittance = sampleTransmittance( transmittance, distance, up.dot( sunDirection ) );
				const shadow = atmosphereRaySphereNearest( position, sunDirection, GROUND_RADIUS_KM );
				const lit = shadow.greaterThan( 0 ).select( 0, 1 );
				const source = sunTransmittance.mul( lit ).mul( scattering ).mul( ISOTROPIC_PHASE );
				const stepTransmittance = extinction.mul( stepSize ).negate().exp();
				const safeExtinction = max( extinction, vec3( 1e-6 ) );
				const integrated = source.sub( source.mul( stepTransmittance ) ).div( safeExtinction );
				const scattered = scattering.sub( scattering.mul( stepTransmittance ) ).div( safeExtinction );

				luminance.addAssign( throughput.mul( integrated ) );
				gathered.addAssign( throughput.mul( scattered ) );
				throughput.mulAssign( stepTransmittance );

			} );

			If( hitGround, () => {

				const groundPosition = origin.add( rayDirection.mul( limit ) );
				const groundUp = normalize( groundPosition );
				const cosGround = groundUp.dot( sunDirection );
				const sunTransmittance = sampleTransmittance( transmittance, float( GROUND_RADIUS_KM ), cosGround );

				luminance.addAssign( sunTransmittance.mul( throughput ).mul( max( cosGround, 0 ) ).mul( albedo ).div( Math.PI ) );

			} );

			luminanceSum.addAssign( luminance.mul( SPHERE_WEIGHT ) );
			gatheredSum.addAssign( gathered.mul( SPHERE_WEIGHT ) );

		} );

		const secondOrder = luminanceSum.mul( ISOTROPIC_PHASE );
		const higherOrder = gatheredSum.mul( ISOTROPIC_PHASE );

		textureStore( texture, globalId.xy, vec4( secondOrder.div( float( 1 ).sub( higherOrder ) ), 1 ) ).toWriteOnly();

	} )().computeKernel( [ 8, 8, 1 ] );

	compute.dispatchSize = [ MULTI_SCATTER_SIZE / 8, MULTI_SCATTER_SIZE / 8, 1 ];
	compute.setName( 'Atmosphere.MultiScattering' );

	return { texture, compute: compute as ComputeNode };

}

/**
 * 采样透射表。`r` 为到地心的距离，`mu` 为光线与当地天顶的余弦。
 *
 * @param transmittance 透射表
 * @param radius 到地心的距离，单位 km
 * @param mu 天顶角余弦
 * @returns 三通道透射，1 为无衰减
 */
export function sampleTransmittance( transmittance: StorageTexture, radius: Node< 'float' >, mu: Node< 'float' > ): Node< 'vec3' > {

	return textureLevel( transmittance, atmosphereTransmittanceUV( radius, mu ), float( 0 ) ).rgb as Node< 'vec3' >;

}

/**
 * 采样多次散射表。
 *
 * 横轴是太阳天顶角余弦，纵轴是到地心距离在 `[Rg, Rt]` 上的归一化高度。
 * 纹素中心内缩 `0.5/尺寸`，和写入时的展开互为逆映射。
 *
 * @param multiScattering 多次散射表
 * @param radius 到地心的距离，单位 km
 * @param cosSun 采样点天顶与太阳方向的余弦
 * @returns `L / (1 - fms)`，未乘太阳照度
 */
export function sampleMultiScattering( multiScattering: StorageTexture, radius: Node< 'float' >, cosSun: Node< 'float' > ): Node< 'vec3' > {

	const uv = vec2(
		cosSun.mul( 0.5 ).add( 0.5 ),
		radius.sub( GROUND_RADIUS_KM ).div( ATMOSPHERE_RADIUS_KM - GROUND_RADIUS_KM ),
	).mul( ( MULTI_SCATTER_SIZE - 1 ) / MULTI_SCATTER_SIZE ).add( 0.5 / MULTI_SCATTER_SIZE );

	return textureLevel( multiScattering, uv, float( 0 ) ).rgb as Node< 'vec3' >;

}
