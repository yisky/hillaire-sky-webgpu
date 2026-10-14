/**
 * 天空颜色：天空视图，加上太阳盘、月亮、恒星和银河。
 *
 * 太阳盘不写入天空视图表。角半径 `0.004675×1.15` rad，临边变暗 `1 - 0.6(1-μ)`，
 * 乘该方向的大气透射，强度 `2500`（物理值约 `1.6e5`，为 fp16 钳位）。
 * `dir.y` 从 `-0.02` 到 `0` 时日盘平滑消失。
 * 月亮方向由宿主写入，角半径 `0.0048` rad。
 * 恒星是立方体面上一格一颗候选，160 格，星等服从幂律，沿银河带更密。
 * 银河极轴是 `normalize(0.3, 0.2, 1)`。`starIntensity` 在太阳落到地平线下时从 0 开到 1。
 */

import { StorageTexture, type Node, type NodeBuilder } from 'three/webgpu';
import {
	Fn,
	abs,
	acos,
	cameraPosition,
	clamp,
	dot,
	exp,
	float,
	floor,
	fract,
	length,
	log2,
	max,
	mix,
	normalize,
	positionWorld,
	pow,
	sign,
	sin,
	smoothstep,
	sqrt,
	vec2,
	vec3,
} from 'three/tsl';

import { SUN_ANGULAR_RADIUS, SUN_DISK_INTENSITY } from '../constants';
import { sampleTransmittance } from '../atmosphere/multiScattering';
import { sampleSkyView } from '../atmosphere/skyView';
import { sampleSkyPro, type SkyProView } from '../clouds/skypro';

/** 立方体每个半面的格子数。一格至多一颗星。 */
const STAR_CELLS = 160;

/** 星点高斯的标准差，单位是格子。`0.1` 大约一像素。 */
const STAR_SIGMA = 0.1;

/** 银河带的极轴。带面是与它垂直的大圆。 */
const MILKY_WAY_POLE = vec3( 0.3, 0.2, 1 ).normalize();

/** 月亮角半径，单位弧度。比太阳盘略大。 */
const MOON_ANGULAR_RADIUS = 0.0048;

/**
 * 无正弦的三维哈希，Hoskins 的 `hash13`。
 *
 * 大整数格子上仍保持精度，用来决定这一格有没有星、星等和颜色。
 *
 * @param point 格子坐标
 * @returns `[0, 1)` 的均匀数
 */
const hash13 = Fn( ( inputs: readonly [ Node< 'vec3' > ], _builder: NodeBuilder ) => {

	const p3 = fract( inputs[ 0 ].mul( vec3( 0.1031, 0.1030, 0.0973 ) ) ).toVar();

	p3.addAssign( dot( p3, p3.yzx.add( 33.33 ) ) );

	return fract( p3.x.add( p3.y ).mul( p3.z ) );

} );

/**
 * 太阳盘亮度。
 *
 * `r` 是角距离除以角半径。`r < 0.9` 为盘内，`r > 1` 为盘外，中间平滑过渡。
 * `μ = sqrt(1-r²)` 是盘面法线余弦，临边变暗把它压到 `0.4`。
 * 透射取观察高度沿 `dir.y` 走到大气外的值，所以低太阳会变红。
 *
 * @param direction 观察方向，单位向量
 * @param sunDirection 世界空间太阳方向
 * @param transmittance 透射表
 * @param viewHeight 观察点到地心的距离，单位 km
 * @returns 盘面亮度。不在盘上时为 0
 */
function sunDisk( direction: Node< 'vec3' >, sunDirection: Node< 'vec3' >, transmittance: StorageTexture, viewHeight: Node< 'float' > ): Node< 'vec3' > {

	const angle = acos( clamp( dot( direction, sunDirection ), - 1, 1 ) );
	const radius = angle.div( SUN_ANGULAR_RADIUS );
	const mask = float( 1 ).sub( smoothstep( 0.9, 1, radius ) );
	const mu = sqrt( max( float( 1 ).sub( radius.mul( radius ) ), 0 ) );
	const limb = float( 1 ).sub( float( 0.6 ).mul( float( 1 ).sub( mu ) ) );
	const transmitted = sampleTransmittance( transmittance, viewHeight, direction.y );
	const above = smoothstep( - 0.02, 0, direction.y );

	return transmitted.mul( mask ).mul( limb ).mul( SUN_DISK_INTENSITY ).mul( above );

}

/**
 * 恒星和银河弥散光。
 *
 * 先选立方体面，再在该面的格子里用哈希放一颗候选星。
 * 银河带上出现概率更高。亮度 `u^-0.8`，亮星的光斑也更大。
 * 太阳高于地平线约 6° 时最亮的星开始出现，低于约 16° 时暗星也出来。
 * 靠近地平线再乘一段消光。
 *
 * @param direction 观察方向，单位向量
 * @param sunDirection 世界空间太阳方向。只用它的高度决定可见星等
 * @param starIntensity 夜深，`0` 为白天，`1` 为全夜
 * @param time 累计时间，单位秒，驱动闪烁
 * @returns 星光加银河，线性 RGB
 */
function stars( direction: Node< 'vec3' >, sunDirection: Node< 'vec3' >, starIntensity: Node< 'float' >, time: Node< 'float' > ): Node< 'vec3' > {

	const absolute = abs( direction );
	const onX = absolute.x.greaterThan( absolute.y ).and( absolute.x.greaterThan( absolute.z ) );
	const onY = absolute.y.greaterThan( absolute.z );
	const face = onX.select( sign( direction.x ).add( 2 ), onY.select( sign( direction.y ).add( 5 ), sign( direction.z ).add( 8 ) ) );
	const uv = onX.select(
		direction.yz.div( absolute.x ),
		onY.select( direction.xz.div( absolute.y ), direction.xy.div( absolute.z ) ),
	).mul( STAR_CELLS );
	const cell = vec3( floor( uv ), face );
	const occurrence = hash13( cell );
	const latitude = dot( direction, MILKY_WAY_POLE ).mul( 4 );
	const band = exp( latitude.mul( latitude ).negate() );
	const present = occurrence.lessThan( band.mul( 0.035 ).add( 0.025 ) );
	const rank = max( hash13( cell.add( 7.7 ) ), 2e-4 );
	const magnitude = log2( rank ).mul( 0.602 ).add( 6.5 );
	const darkness = float( 1 ).sub( smoothstep( - 0.28, - 0.1, sunDirection.y ) );
	const visible = smoothstep( magnitude.sub( 0.6 ), magnitude.add( 0.6 ), darkness.mul( 7.5 ).sub( 1 ) ).mul( present.select( 1, 0 ) );
	const point = floor( uv ).add( vec2( hash13( cell.add( 3.1 ) ), hash13( cell.add( 5.7 ) ) ).mul( 0.4 ).add( 0.3 ) ).div( STAR_CELLS );
	const starDirection = normalize( onX.select(
		vec3( sign( direction.x ), point ),
		onY.select( vec3( point.x, sign( direction.y ), point.y ), vec3( point, sign( direction.z ) ) ),
	) );
	const distance = length( direction.sub( starDirection ) ).mul( STAR_CELLS );
	const flux = pow( rank, - 0.8 );
	const size = log2( flux ).mul( 0.08 ).add( 1 );
	const spread = exp( distance.mul( distance ).div( size.mul( size ) ).mul( - 0.5 / ( STAR_SIGMA * STAR_SIGMA ) ) ).div( size.mul( size ) );
	const twinkle = sin( time.mul( hash13( cell.add( 13.3 ) ).mul( 9 ).add( 5 ) ).add( occurrence.mul( 60 ) ) )
		.mul( mix( 0.18, 0.06, clamp( direction.y.mul( 2 ), 0, 1 ) ) )
		.add( 1 );
	const color = mix( vec3( 1, 0.8, 0.6 ), vec3( 0.75, 0.85, 1 ), hash13( cell.add( 17 ) ) ).mul( 0.5 ).add( 0.5 );
	const star = color.mul( spread.mul( flux ).mul( visible ).mul( twinkle ).mul( 0.0075 ) );
	const glow = vec3( 0.55, 0.6, 0.75 ).mul( band.mul( darkness ).mul( 0.0035 ) );

	return star.add( glow ).mul( starIntensity ).mul( smoothstep( 0, 0.2, direction.y ) );

}

/**
 * 月盘。
 *
 * 颜色偏冷白，亮度 `3`，并乘夜深。贴地平线以下平滑消失。
 *
 * @param direction 观察方向，单位向量
 * @param moonDirection 世界空间月亮方向
 * @param starIntensity 夜深
 * @returns 月盘亮度
 */
function moonDisk( direction: Node< 'vec3' >, moonDirection: Node< 'vec3' >, starIntensity: Node< 'float' > ): Node< 'vec3' > {

	const angle = acos( clamp( dot( direction, moonDirection ), - 1, 1 ) );
	const radius = angle.div( MOON_ANGULAR_RADIUS );
	const mask = float( 1 ).sub( smoothstep( 0.92, 1, radius ) );
	const above = smoothstep( - 0.02, 0.02, direction.y );

	return vec3( 0.9, 0.92, 1 ).mul( mask ).mul( 3 ).mul( starIntensity ).mul( above );

}

/**
 * 月光铺开的夜空和月晕。
 *
 * 底色约 `(0.005, 0.0068, 0.0105)`，靠近地平线更亮。月晕是两段指数。
 * 月亮本身落到地平线附近时整项减弱。
 *
 * @param direction 观察方向，单位向量
 * @param moonDirection 世界空间月亮方向
 * @param starIntensity 夜深
 * @returns 夜空亮度
 */
function moonlitSky( direction: Node< 'vec3' >, moonDirection: Node< 'vec3' >, starIntensity: Node< 'float' > ): Node< 'vec3' > {

	const angle = acos( clamp( dot( direction, moonDirection ), - 1, 1 ) );
	const aureole = exp( angle.mul( - 14 ) ).mul( 2.4 ).add( exp( angle.mul( - 2.5 ) ).mul( 0.9 ) );
	const horizon = mix( 1.7, 1, clamp( direction.y.mul( 3 ), 0, 1 ) );
	const moonUp = smoothstep( - 0.05, 0.15, moonDirection.y );

	return vec3( 0.005, 0.0068, 0.0105 ).mul( horizon.add( aureole ) ).mul( starIntensity ).mul( moonUp );

}

/**
 * 地平线以下不用大气表。那一段射线立刻撞上地面，积分是黑的。
 * 改用地平线同方位的天空颜色，越往下越暗。颜色跟着太阳走。
 */
function withGround(
	color: Node< 'vec3' >,
	skyView: StorageTexture,
	viewHeight: Node< 'float' >,
	sun: Node,
	direction: Node< 'vec3' >,
): Node< 'vec3' > {

	const horizonDir = normalize( vec3( direction.x.add( 1e-5 ), max( direction.y, 0 ), direction.z ) );
	const horizon = sampleSkyView( skyView, viewHeight, sun, horizonDir );
	const depth = smoothstep( 0, - 0.5, direction.y );
	const ground = horizon.mul( mix( 1, 0.4, depth ) );

	return direction.y.lessThan( 0 ).select( ground, color );

}

/**
 * 构造天空颜色节点。
 *
 * 视线是片元世界坐标减去相机位置。天空盒对齐相机后，这个方向就是观察方向。
 * 叠加顺序是大气、月光夜空、恒星、月盘、太阳盘。
 *
 * @param skyView 天空视图表
 * @param transmittance 透射表，只用于太阳盘
 * @param viewHeight 写入天空视图时的地心距离，单位 km
 * @param sunDirection 世界空间太阳方向
 * @param moonDirection 世界空间月亮方向
 * @param starIntensity 夜深，`0..1`
 * @param time 累计秒数
 * @returns 天空亮度，线性 RGB
 */
export function createSkyColorNode(
	skyView: StorageTexture,
	transmittance: StorageTexture,
	viewHeight: Node< 'float' >,
	sunDirection: Node,
	moonDirection: Node,
	starIntensity: Node< 'float' >,
	time: Node< 'float' >,
): Node< 'vec3' > {

	const direction = normalize( positionWorld.sub( cameraPosition ) ) as Node< 'vec3' >;
	const sun = sunDirection as Node< 'vec3' >;
	const moon = moonDirection as Node< 'vec3' >;
	const sky = sampleSkyView( skyView, viewHeight, sun, direction );

	return withGround(
		sky
			.add( moonlitSky( direction, moon, starIntensity ) )
			.add( stars( direction, sun, starIntensity, time ) )
			.add( moonDisk( direction, moon, starIntensity ) )
			.add( sunDisk( direction, sun, transmittance, viewHeight ) ),
		skyView,
		viewHeight,
		sun,
		direction,
	);

}

/** 环境贴图里的星光比例。`0.08` 避免亮星在粗糙反射里闪。 */
const STAR_REFLECTION = 0.08;

/**
 * 给 SkyProbe 用的天空。大气和月光夜空保留，没有太阳盘和月亮盘，恒星乘 {@link STAR_REFLECTION}。
 * 视锥里的云不在这里。
 */
export function createReflectionSkyColor(
	skyView: StorageTexture,
	viewHeight: Node< 'float' >,
	sunDirection: Node,
	moonDirection: Node,
	starIntensity: Node< 'float' >,
	time: Node< 'float' >,
): Node< 'vec3' > {

	const direction = normalize( positionWorld.sub( cameraPosition ) ) as Node< 'vec3' >;
	const sun = sunDirection as Node< 'vec3' >;
	const moon = moonDirection as Node< 'vec3' >;

	return sampleSkyView( skyView, viewHeight, sun, direction )
		.add( moonlitSky( direction, moon, starIntensity ) )
		.add( stars( direction, sun, starIntensity, time ).mul( STAR_REFLECTION ) );

}

/**
 * 晴空叠上视锥里的云。
 *
 * 历史缓冲的 `a` 是不透明度。采样时改成透射：天空、月亮和恒星乘透射，再加上云的内散射。
 * 太阳盘再乘 `T × smoothstep(0.004, 0.04, T)`，避免厚云缝里透出日盘。
 * 缓冲还没写，或视线落到视锥外时，透射是 1。
 *
 * @param skyView 天空视图表
 * @param transmittance 透射表，只用于太阳盘
 * @param viewHeight 观察点到地心的距离，单位 km
 * @param sunDirection 世界空间太阳方向
 * @param moonDirection 世界空间月亮方向
 * @param starIntensity 夜深，`0..1`
 * @param time 累计秒数
 * @param view 本实例的云历史，`vec4(内散射, 不透明度)`
 * @param camera 写这张缓冲时的相机
 * @returns 天空亮度，线性 RGB
 */
export function createSkyColorWithClouds(
	skyView: StorageTexture,
	transmittance: StorageTexture,
	viewHeight: Node< 'float' >,
	sunDirection: Node,
	moonDirection: Node,
	starIntensity: Node< 'float' >,
	time: Node< 'float' >,
	view: StorageTexture,
	camera: SkyProView,
): Node< 'vec3' > {

	const direction = normalize( positionWorld.sub( cameraPosition ) ) as Node< 'vec3' >;
	const sun = sunDirection as Node< 'vec3' >;
	const moon = moonDirection as Node< 'vec3' >;
	const sky = sampleSkyView( skyView, viewHeight, sun, direction )
		.add( moonlitSky( direction, moon, starIntensity ) )
		.add( stars( direction, sun, starIntensity, time ) )
		.add( moonDisk( direction, moon, starIntensity ) );
	const disk = sunDisk( direction, sun, transmittance, viewHeight );
	const cloud = sampleSkyPro( view, camera, direction );
	const diskCover = cloud.a.mul( smoothstep( 0.004, 0.04, cloud.a ) );

	return withGround(
		sky.mul( cloud.a ).add( disk.mul( diskCover ) ).add( cloud.rgb ),
		skyView,
		viewHeight,
		sun,
		direction,
	);

}
