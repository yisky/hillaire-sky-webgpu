/**
 * 天空视图查找表，以及从观察方向采样它。
 *
 * 表里是不含太阳盘的天空亮度，Hillaire EGSR 2020 第 5.3 节。
 * 地平线张角随观察高度变化：β = acos(sqrt(r² - Rg²) / r)，天顶到地平线的天顶角是 π - β。
 * `v < 0.5` 用平方根铺天顶到地平线，`v > 0.5` 铺地平线以下。
 * 横轴是相对太阳的方位，`u = sqrt((1 - cos) / 2)`，太阳在 `u = 0`。
 * 沿视线 32 步，步长按参数的平方拉开，样本落在步长的 30% 处。
 * 瑞利相位是 `(3/16π)(1 + cos²θ)`。米氏相位是 Cornette-Shanks，`g` 默认 0.8。
 * 源项是 `透射 × 地球影 × 相位散射 + Ψms × σs`，解析积分 `(S - S·e^-σΔt) / σ`，最后乘场景照度。
 * 距离单位为千米。
 */

import {
	ClampToEdgeWrapping,
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
	Return,
	acos,
	clamp,
	cos,
	dot,
	float,
	globalId,
	max,
	mix,
	normalize,
	sin,
	sqrt,
	textureLevel,
	textureStore,
	uniform,
	uint,
	vec2,
	vec3,
	vec4,
} from 'three/tsl';

import {
	ATMOSPHERE_RADIUS_KM,
	GROUND_RADIUS_KM,
	SKY_VIEW_HEIGHT,
	SKY_VIEW_WIDTH,
	SUN_ILLUMINANCE,
} from '../constants';
import { atmosphereMedium, atmosphereRaySphereNearest } from './medium';
import { sampleMultiScattering, sampleTransmittance } from './multiScattering';

/** 天空视图的步数。参数平方后，近处步长短、远处步长长。 */
const SKY_VIEW_STEPS = 32;

/** 瑞利相位的系数 `3 / 16π`。 */
const RAYLEIGH_PHASE = 3 / ( 16 * Math.PI );

/** Cornette-Shanks 相位的系数 `3 / 8π`。 */
const MIE_PHASE = 3 / ( 8 * Math.PI );

/**
 * 创建天空视图表。
 *
 * 分辨率 `192×108`，`rgba16float`。高度不是 8 的倍数，分发时多出的线程直接返回。
 * `viewHeight` 是到地心的距离，单位 km。`mieG` 只进入相位，不进入另外两张表。
 *
 * @param transmittance 透射表
 * @param multiScattering 多次散射表
 * @param rayleighScale 瑞利倍数
 * @param mieScale 米氏倍数
 * @param ozoneScale 臭氧倍数
 * @param sunDirection 世界空间太阳方向。积分只用它的高度角，方位在采样时再对齐
 * @returns 纹理、compute，以及本表使用的高度和 `mieG`
 */
export function createSkyViewLut(
	transmittance: StorageTexture,
	multiScattering: StorageTexture,
	rayleighScale: Node< 'float' >,
	mieScale: Node< 'float' >,
	ozoneScale: Node< 'float' >,
	sunDirection: Node,
) {

	const viewHeight = uniform( GROUND_RADIUS_KM + 0.002 );
	const mieG = uniform( 0.8 );
	const texture = new StorageTexture( SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT );
	texture.name = 'atmoSkyView';
	texture.type = HalfFloatType;
	texture.magFilter = LinearFilter;
	texture.minFilter = LinearFilter;
	texture.wrapS = ClampToEdgeWrapping;
	texture.wrapT = ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.flipY = false;
	( texture as StorageTexture & { mipmapsAutoUpdate: boolean } ).mipmapsAutoUpdate = false;

	const stepLoop = { start: 0, end: SKY_VIEW_STEPS, type: 'int' as const, condition: '<', name: 'step' };

	const compute = Fn( () => {

		If( globalId.x.greaterThanEqual( uint( SKY_VIEW_WIDTH ) ).or( globalId.y.greaterThanEqual( uint( SKY_VIEW_HEIGHT ) ) ), () => {

			Return();

		} );

		const texel = vec2( float( globalId.x ), float( globalId.y ) ).add( 0.5 ).div( vec2( SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT ) );
		const u = texel.x.sub( 0.5 / SKY_VIEW_WIDTH ).mul( SKY_VIEW_WIDTH / ( SKY_VIEW_WIDTH - 1 ) ).toVar();
		const v = texel.y.sub( 0.5 / SKY_VIEW_HEIGHT ).mul( SKY_VIEW_HEIGHT / ( SKY_VIEW_HEIGHT - 1 ) ).toVar();
		const horizon = sqrt( max( viewHeight.mul( viewHeight ).sub( GROUND_RADIUS_KM * GROUND_RADIUS_KM ), 0 ) );
		const beta = acos( horizon.div( viewHeight ) );
		const zenithToHorizon = float( Math.PI ).sub( beta );
		const above = float( 1 ).sub( v.mul( 2 ) );
		const viewZenithAbove = zenithToHorizon.mul( float( 1 ).sub( above.mul( above ) ) );
		const below = v.mul( 2 ).sub( 1 );
		const viewZenith = v.lessThan( 0.5 ).select( viewZenithAbove, zenithToHorizon.add( beta.mul( below.mul( below ) ) ) ).toVar();
		const cosView = cos( viewZenith );
		const sinView = sin( viewZenith );
		const lightViewCos = u.mul( u ).mul( 2 ).sub( 1 ).negate();
		const lightViewSin = sqrt( max( float( 1 ).sub( lightViewCos.mul( lightViewCos ) ), 0 ) );
		const sunNode = sunDirection as Node< 'vec3' >;
		const sunCos = sunNode.y;
		const sunSin = sqrt( max( float( 1 ).sub( sunCos.mul( sunCos ) ), 0 ) );
		const sun = vec3( sunSin, sunCos, 0 );
		const rayDirection = vec3( sinView.mul( lightViewCos ), cosView, sinView.mul( lightViewSin ) );
		const origin = vec3( 0, viewHeight, 0 );
		const tGround = atmosphereRaySphereNearest( origin, rayDirection, GROUND_RADIUS_KM ).toVar();
		const tTop = atmosphereRaySphereNearest( origin, rayDirection, ATMOSPHERE_RADIUS_KM ).toVar();
		const limit = tGround.greaterThan( 0 ).select( tGround, tTop ).toVar();
		const cosTheta = dot( rayDirection, sun ).toVar();
		const rayPhase = float( RAYLEIGH_PHASE ).mul( cosTheta.mul( cosTheta ).add( 1 ) );
		const anisotropy = mieG.mul( mieG );
		const mieDenominator = max( anisotropy.add( 1 ).sub( mieG.mul( cosTheta ).mul( 2 ) ), 1e-4 ).pow( 1.5 );
		const miePhase = float( MIE_PHASE ).mul( float( 1 ).sub( anisotropy ).mul( cosTheta.mul( cosTheta ).add( 1 ) ) ).div( anisotropy.add( 2 ).mul( mieDenominator ) );
		const throughput = vec3( 1 ).toVar();
		const luminance = vec3( 0 ).toVar();

		Loop( stepLoop, ( input ) => {

			const index = float( ( input as unknown as { step: Node< 'int' > } ).step );
			const t0 = index.div( SKY_VIEW_STEPS );
			const t1 = index.add( 1 ).div( SKY_VIEW_STEPS );
			const near = t0.mul( t0 ).mul( limit );
			const far = t1.mul( t1 ).mul( limit );
			const traveled = mix( near, far, 0.3 );
			const stepSize = far.sub( near );
			const position = origin.add( rayDirection.mul( traveled ) );
			const distance = position.length().toVar();
			const up = position.div( distance );
			const medium = atmosphereMedium( distance.sub( GROUND_RADIUS_KM ), rayleighScale, mieScale, ozoneScale );
			const rayScat = medium.get( 'rayScat' ) as Node< 'vec3' >;
			const mieScat = medium.get( 'mieScat' ) as Node< 'float' >;
			const scattering = medium.get( 'scattering' ) as Node< 'vec3' >;
			const extinction = medium.get( 'extinction' ) as Node< 'vec3' >;
			const cosSun = up.dot( sun );
			const sunTransmittance = sampleTransmittance( transmittance, distance, cosSun );
			const shadow = atmosphereRaySphereNearest( position, sun, GROUND_RADIUS_KM );
			const lit = shadow.greaterThan( 0 ).select( 0, 1 );
			const multiple = sampleMultiScattering( multiScattering, distance, cosSun );
			const phaseScattering = rayScat.mul( rayPhase ).add( vec3( mieScat.mul( miePhase ) ) );
			const source = sunTransmittance.mul( lit ).mul( phaseScattering ).add( multiple.mul( scattering ) );
			const stepTransmittance = extinction.mul( stepSize ).negate().exp();
			const integrated = source.sub( source.mul( stepTransmittance ) ).div( max( extinction, vec3( 1e-6 ) ) );

			luminance.addAssign( throughput.mul( integrated ) );
			throughput.mulAssign( stepTransmittance );

		} );

		textureStore( texture, globalId.xy, vec4( luminance.mul( SUN_ILLUMINANCE ), 1 ) ).toWriteOnly();

	} )().computeKernel( [ 8, 8, 1 ] );

	compute.dispatchSize = [ Math.ceil( SKY_VIEW_WIDTH / 8 ), Math.ceil( SKY_VIEW_HEIGHT / 8 ), 1 ];
	compute.setName( 'Atmosphere.SkyView' );

	return { texture, compute: compute as ComputeNode, viewHeight, mieG };

}

/**
 * 按观察方向采样天空视图。
 *
 * 方位相对太阳：太阳所在的水平方向是 `u = 0`。高度先换成天顶角，
 * 再按写入时的地平线张角映回 `v`。结果已乘场景照度，不含太阳盘。
 *
 * @param skyView 天空视图表
 * @param viewHeight 观察点到地心的距离，单位 km，须与写入这张表时相同
 * @param sunDirection 世界空间太阳方向
 * @param direction 世界空间观察方向，单位向量
 * @returns 天空亮度，线性 RGB
 */
export function sampleSkyView(
	skyView: StorageTexture,
	viewHeight: Node< 'float' >,
	sunDirection: Node,
	direction: Node,
): Node< 'vec3' > {

	const horizon = sqrt( max( viewHeight.mul( viewHeight ).sub( GROUND_RADIUS_KM * GROUND_RADIUS_KM ), 0 ) );
	const beta = acos( horizon.div( viewHeight ) );
	const zenithToHorizon = float( Math.PI ).sub( beta );
	const viewDirection = direction as Node< 'vec3' >;
	const sunNode = sunDirection as Node< 'vec3' >;
	const viewZenith = acos( clamp( viewDirection.y, - 1, 1 ) );
	const above = float( 1 ).sub( sqrt( max( float( 1 ).sub( viewZenith.div( zenithToHorizon ) ), 0 ) ) ).mul( 0.5 );
	const below = sqrt( max( viewZenith.sub( zenithToHorizon ).div( beta ), 0 ) ).mul( 0.5 ).add( 0.5 );
	const v = viewZenith.lessThan( zenithToHorizon ).select( above, below );
	const sunHorizontal = normalize( vec2( sunNode.x, sunNode.z ).add( vec2( 1e-5, 0 ) ) );
	const viewHorizontal = normalize( vec2( viewDirection.x, viewDirection.z ).add( vec2( 1e-5, 0 ) ) );
	const u = sqrt( clamp( dot( sunHorizontal, viewHorizontal ).mul( - 0.5 ).add( 0.5 ), 0, 1 ) );
	const uv = vec2(
		u.mul( ( SKY_VIEW_WIDTH - 1 ) / SKY_VIEW_WIDTH ).add( 0.5 / SKY_VIEW_WIDTH ),
		v.mul( ( SKY_VIEW_HEIGHT - 1 ) / SKY_VIEW_HEIGHT ).add( 0.5 / SKY_VIEW_HEIGHT ),
	);

	return textureLevel( skyView, uv, float( 0 ) ).rgb as Node< 'vec3' >;

}
