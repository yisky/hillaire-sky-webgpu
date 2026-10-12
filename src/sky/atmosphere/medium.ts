/**
 * 大气介质、透射参数化和射线与球的交点。
 *
 * 密度和消光对应 Hillaire EGSR 2020 第 4 节，距离单位为千米。
 * 透射纹理坐标是 Bruneton 2008 的 `(ρ, d)`，论文第 5.2 节采用。
 * 公式与 Tidewater `Atmosphere.js` 的 `atmosphereMedium`、
 * `atmosphereTransmittanceUV`、`atmosphereRaySphereNearest` 相同。
 */

import { Fn, abs, dot, float, max, select, sqrt, struct, vec2, vec3 } from 'three/tsl';
import { NodeBuilder, type Node } from 'three/webgpu';

import { ATMOSPHERE_RADIUS_KM, GROUND_RADIUS_KM, TRANSMITTANCE_HEIGHT, TRANSMITTANCE_WIDTH } from '../constants';

/**
 * 瑞利散射系数，单位 km^-1。
 *
 * 来自论文表 1，已从 m^-1 换成 km^-1：`(5.802, 13.558, 33.1) × 10^-3`。
 * 蓝通道最大，所以长路径先失去蓝色。
 */
const RAYLEIGH_SCATTERING = vec3( 5.802e-3, 13.558e-3, 33.1e-3 );

/** 米氏散射系数，单位 km^-1。论文表 1 的 `3.996 × 10^-3`。 */
const MIE_SCATTERING = 3.996e-3;

/**
 * 米氏消光系数，单位 km^-1。
 *
 * Bruneton / Unreal 参考实现用 `4.440e-3`，吸收约为 `0.444e-3`。
 * 不把论文表 1 的 `4.40e-6 m^-1` 那一列整列当成吸收。
 */
const MIE_EXTINCTION = 4.440e-3;

/**
 * 臭氧吸收系数，单位 km^-1。
 *
 * `(0.650, 1.881, 0.085) × 10^-3`。绿通道吸收最强。
 */
const OZONE_ABSORPTION = vec3( 0.650e-3, 1.881e-3, 0.085e-3 );

/** 瑞利标高，单位 km。 */
const RAYLEIGH_SCALE_HEIGHT = 8;

/** 米氏标高，单位 km。 */
const MIE_SCALE_HEIGHT = 1.2;

/** 臭氧帐篷中心高度，单位 km。 */
const OZONE_CENTER_KM = 25;

/** 臭氧帐篷半宽，单位 km。中心处密度为 1，上下 15 km 处为 0。 */
const OZONE_WIDTH_KM = 15;

/**
 * 某一高度上的散射与消光。
 *
 * 瑞利和米氏密度是 `exp(-h / H)`。臭氧是以 25 km 为中心、半宽 15 km 的帐篷，
 * 只吸收、不散射。消光 = 瑞利散射 + 米氏消光 + 臭氧吸收。
 * 散射 = 瑞利散射 + 米氏散射，供天空视图的源项使用。
 *
 * @param hKm 采样点海拔，单位 km。地面为 0
 * @param rayleighScale 瑞利系数倍数，无量纲
 * @param mieScale 米氏散射和消光的共同倍数，无量纲
 * @param ozoneScale 臭氧吸收倍数，无量纲
 * @returns `rayScat`、`mieScat`、`extinction`、`scattering`
 */
const AtmosphereMedium = struct( {
	rayScat: 'vec3',
	mieScat: 'float',
	extinction: 'vec3',
	scattering: 'vec3',
}, 'AtmosphereMedium' );

export const atmosphereMedium = Fn( ( inputs: readonly [ Node< 'float' >, Node< 'float' >, Node< 'float' >, Node< 'float' > ], _builder: NodeBuilder ) => {

	const hKm = inputs[ 0 ];
	const rayleighScale = inputs[ 1 ];
	const mieScale = inputs[ 2 ];
	const ozoneScale = inputs[ 3 ];
	const rayDensity = expNeg( hKm, RAYLEIGH_SCALE_HEIGHT );
	const mieDensity = expNeg( hKm, MIE_SCALE_HEIGHT );
	const ozoneDensity = float( 1 ).sub( abs( hKm.sub( OZONE_CENTER_KM ) ).div( OZONE_WIDTH_KM ) ).max( 0 );
	const rayScat = RAYLEIGH_SCATTERING.mul( rayDensity ).mul( rayleighScale );
	const mieScat = float( MIE_SCATTERING ).mul( mieDensity ).mul( mieScale );
	const mieExt = float( MIE_EXTINCTION ).mul( mieDensity ).mul( mieScale );
	const ozoneAbs = OZONE_ABSORPTION.mul( ozoneDensity ).mul( ozoneScale );

	return AtmosphereMedium(
		rayScat,
		mieScat,
		rayScat.add( mieExt ).add( ozoneAbs ),
		rayScat.add( mieScat ),
	);

} );

/**
 * 把 `(r, μ)` 映到透射表的纹理坐标。
 *
 * `r` 是到地心的距离，`μ` 是光线与当地天顶的余弦。
 * `ρ = sqrt(r² - Rg²)`，`d` 是从该点沿 `μ` 到大气顶的距离。
 * 返回值落在纹素中心：单位坐标先加 `0.5/尺寸`，再乘 `尺寸/(尺寸+1)`，
 * 避免采样插到表外。
 *
 * @param r 到地心的距离，单位 km，范围 `[Rg, Rt]`
 * @param mu 天顶角余弦，范围 `[-1, 1]`
 * @returns 透射表 UV，范围大约在 `(0, 1)`
 */
export const atmosphereTransmittanceUV = Fn( ( inputs: readonly [ Node< 'float' >, Node< 'float' > ], _builder: NodeBuilder ) => {

	const r = inputs[ 0 ];
	const mu = inputs[ 1 ];
	const H = horizonDistance();
	const rho = sqrt( max( r.mul( r ).sub( GROUND_RADIUS_KM * GROUND_RADIUS_KM ), 0 ) );
	const discriminant = r.mul( r ).mul( mu.mul( mu ).sub( 1 ) ).add( ATMOSPHERE_RADIUS_KM * ATMOSPHERE_RADIUS_KM );
	const d = max( r.negate().mul( mu ).add( sqrt( max( discriminant, 0 ) ) ), 0 ) as Node< 'float' >;
	const dMin = float( ATMOSPHERE_RADIUS_KM ).sub( r );
	const dMax = rho.add( H );
	const xMu = d.sub( dMin ).div( dMax.sub( dMin ) );
	const xR = rho.div( H );

	return vec2(
		xMu.add( 0.5 / TRANSMITTANCE_WIDTH ).mul( TRANSMITTANCE_WIDTH / ( TRANSMITTANCE_WIDTH + 1 ) ),
		xR.add( 0.5 / TRANSMITTANCE_HEIGHT ).mul( TRANSMITTANCE_HEIGHT / ( TRANSMITTANCE_HEIGHT + 1 ) ),
	);

} );

/**
 * 射线与以地心为球心的球面的最近正交点。
 *
 * 解 `t² + 2 b t + c = 0`，其中 `b = dot(ro, rd)`，`c = |ro|² - R²`，`rd` 已归一化。
 * 取大于 0 的较小根。没有正解时返回 `-1`。
 *
 * @param ro 射线起点，地心为原点，单位 km
 * @param rd 射线方向，单位向量
 * @param radius 球半径，单位 km
 * @returns 最近的正距离，单位 km。未击中为 `-1`
 */
export const atmosphereRaySphereNearest = Fn( ( inputs: readonly [ Node< 'vec3' >, Node< 'vec3' >, Node< 'float' > ], _builder: NodeBuilder ) => {

	const ro = inputs[ 0 ];
	const rd = inputs[ 1 ];
	const radius = inputs[ 2 ];
	const b = dot( ro, rd );
	const c = dot( ro, ro ).sub( radius.mul( radius ) );
	const discriminant = b.mul( b ).sub( c );
	const root = sqrt( max( discriminant, 0 ) );
	const nearer = b.negate().sub( root );
	const farther = b.negate().add( root );
	const positive = select( nearer.greaterThan( 0 ), nearer, select( farther.greaterThan( 0 ), farther, - 1 ) );

	return select( discriminant.lessThan( 0 ), - 1, positive );

} );

/**
 * `exp(-h / H)`。`H` 是标高，单位 km。
 *
 * @param heightKm 海拔，单位 km
 * @param scaleHeightKm 标高，单位 km
 * @returns 相对海平面的密度，无量纲
 */
function expNeg( heightKm: Node< 'float' >, scaleHeightKm: number ): Node< 'float' > {

	return heightKm.div( scaleHeightKm ).negate().exp();

}

/**
 * 地面视界到大气顶的水平距离 `H = sqrt(Rt² - Rg²)`，单位 km。
 *
 * @returns 该距离
 */
function horizonDistance(): Node< 'float' > {

	return sqrt( float( ATMOSPHERE_RADIUS_KM * ATMOSPHERE_RADIUS_KM - GROUND_RADIUS_KM * GROUND_RADIUS_KM ) );

}
