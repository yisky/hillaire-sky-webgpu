/**
 * 每个天空一份的 SkyPro 积云。
 *
 * 预设是 SkyPro 的 “Partly cloudy”。
 * 半分辨率历史，每帧只追踪 4×4 里的一个槽。形状噪声来自 `public/clouds`。
 * 全景和地面云影不在这里。
 */

import {
	ClampToEdgeWrapping,
	DataTexture,
	LinearFilter,
	NearestFilter,
	NoColorSpace,
	PerspectiveCamera,
	RedFormat,
	RepeatWrapping,
	RGBAFormat,
	StorageTexture,
	UnsignedByteType,
	HalfFloatType,
	Vector2,
	Vector3,
	type Camera,
	type Color,
	type ComputeNode,
	type Node,
	type WebGPURenderer,
} from 'three/webgpu';
import {
	Fn,
	If,
	float,
	fract,
	globalId,
	max,
	min,
	sampler,
	smoothstep,
	storageTexture,
	texture,
	textureLevel,
	textureLoad,
	textureStore,
	uniform,
	uvec2,
	vec2,
	vec3,
	vec4,
	dot,
} from 'three/tsl';

import { skyproBounds, skyproResolve, skyproTrace, skyproWeather } from './skypro-wgsl';

const WIND = new Vector2( 0.35, 0.94 ).normalize();
const HISTORY_WEIGHT = 0.9;

/** 采样历史缓冲时用的相机。方向是世界空间单位向量。 */
export type SkyProView = {
	right: Node< 'vec3' >;
	up: Node< 'vec3' >;
	fwd: Node< 'vec3' >;
	/** `x` 是水平切向，`y` 是垂直切向。 */
	tan: Node< 'vec2' >;
	valid: Node< 'float' >;
};

type Pose = {
	position: Vector3;
	right: Vector3;
	up: Vector3;
	forward: Vector3;
	tanY: number;
	aspect: number;
};

/**
 * 把观察方向投回云缓冲。缓冲里 `a` 是不透明度，返回值的 `a` 是透射。
 *
 * @param view 半分辨率历史，`vec4(内散射, 不透明度)`
 * @param camera 写这张缓冲时的相机
 * @param direction 观察方向，单位向量
 */
export function sampleSkyPro( view: StorageTexture, camera: SkyProView, direction: Node< 'vec3' > ): Node< 'vec4' > {

	const ahead = max( dot( direction, camera.fwd ), float( 1e-4 ) );
	const ndc = vec2( dot( direction, camera.right ), dot( direction, camera.up ) ).div( camera.tan.mul( ahead ) );
	const uv = vec2( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ) );
	const inside = camera.valid.greaterThan( 0.5 )
		.and( ahead.greaterThan( 0.01 ) )
		.and( uv.x.greaterThanEqual( 0 ) )
		.and( uv.y.greaterThanEqual( 0 ) )
		.and( uv.x.lessThanEqual( 1 ) )
		.and( uv.y.lessThanEqual( 1 ) );
	const sample = textureLevel( view, uv, float( 0 ) ) as Node< 'vec4' >;
	const above = smoothstep( - 0.05, - 0.03, direction.y );
	const rgb = max( sample.rgb, vec3( 0, 0, 0 ) ).mul( above );
	const transmittance = float( 1 ).sub( min( sample.a, float( 1 ) ).mul( above ) );
	const hit = vec4( rgb, transmittance );

	return inside.select( hit, vec4( 0, 0, 0, 1 ) ) as Node< 'vec4' >;

}

function makeHalf( width: number, height: number, name: string ): StorageTexture {

	const texture = new StorageTexture( width, height );

	texture.name = name;
	texture.type = HalfFloatType;
	texture.format = RGBAFormat;
	texture.magFilter = LinearFilter;
	texture.minFilter = LinearFilter;
	texture.wrapS = ClampToEdgeWrapping;
	texture.wrapT = ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.colorSpace = NoColorSpace;
	( texture as StorageTexture & { mipmapsAutoUpdate: boolean } ).mipmapsAutoUpdate = false;

	return texture;

}

function makeByte( width: number, height: number, name: string, repeat: boolean ): StorageTexture {

	const texture = new StorageTexture( width, height );

	texture.name = name;
	texture.type = UnsignedByteType;
	texture.format = RGBAFormat;
	texture.magFilter = LinearFilter;
	texture.minFilter = LinearFilter;
	texture.wrapS = repeat ? RepeatWrapping : ClampToEdgeWrapping;
	texture.wrapT = repeat ? RepeatWrapping : ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.colorSpace = NoColorSpace;
	( texture as StorageTexture & { mipmapsAutoUpdate: boolean } ).mipmapsAutoUpdate = false;

	return texture;

}

/** 把 64³ 体数据按 8×8 个切片铺成一张二维图。WebGPU 不能给三维纹理加渲染用途。 */
function packNoiseAtlas( volume: Uint8Array, width: number, height: number, depth: number ): Uint8Array {

	const tiles = 8;
	const atlasWidth = width * tiles;
	const packed = new Uint8Array( atlasWidth * height * tiles * 4 );

	for ( let z = 0; z < depth; z ++ ) {

		const tileX = z % tiles;
		const tileY = Math.floor( z / tiles );

		for ( let y = 0; y < height; y ++ ) {

			for ( let x = 0; x < width; x ++ ) {

				const source = ( ( z * height + y ) * width + x ) * 4;
				const dest = ( ( tileY * height + y ) * atlasWidth + tileX * width + x ) * 4;

				packed[ dest ] = volume[ source ];
				packed[ dest + 1 ] = volume[ source + 1 ];
				packed[ dest + 2 ] = volume[ source + 2 ];
				packed[ dest + 3 ] = volume[ source + 3 ];

			}

		}

	}

	return packed;

}

/**
 * 一个天空实例的积云场。
 *
 * 天气图和历史属于这一份。形状噪声和蓝噪声在第一次 `update` 时从文件读入。
 */
export class SkyProField {

	/** 屏幕采样的历史。`a` 是不透明度。 */
	readonly viewClouds: StorageTexture;

	/** 和 {@link SkyProField.viewClouds} 一起交给合成。 */
	readonly viewCamera: SkyProView;

	private readonly weather = makeByte( 1024, 1024, 'skyproWeather', true );

	private readonly bounds = makeByte( 64, 64, 'skyproBounds', false );

	private readonly source = makeHalf( 4, 4, 'skyproSource' );

	private readonly sourceMeta = makeHalf( 4, 4, 'skyproSourceMeta' );

	private readonly scratch = makeHalf( 4, 4, 'skyproScratch' );

	private readonly scratchMeta = makeHalf( 4, 4, 'skyproScratchMeta' );

	private readonly historyMeta: StorageTexture;

	private noise: DataTexture | null = null;

	private blue: DataTexture | null = null;

	private noiseReady = false;

	private built = false;

	private weatherDone = false;

	private historyValid = false;

	private frameIndex = 0;

	private windX = 0;

	private windZ = 0;

	private evolution = 0;

	private readonly drawing = new Vector2();

	private viewWidth = 0;

	private viewHeight = 0;

	private sourceWidth = 1;

	private sourceHeight = 1;

	private historyWidth = 4;

	private historyHeight = 4;

	private readonly slotNode = uniform( new Vector2() );

	private readonly historySizeNode = uniform( new Vector2( 4, 4 ) );

	private readonly sourceSizeNode = uniform( new Vector2( 1, 1 ) );

	private readonly originNode = uniform( new Vector3() );

	private readonly rightNode = uniform( new Vector3( 1, 0, 0 ) );

	private readonly upNode = uniform( new Vector3( 0, 1, 0 ) );

	private readonly fwdNode = uniform( new Vector3( 0, 0, - 1 ) );

	private readonly tanYNode = uniform( 1 );

	private readonly aspectNode = uniform( 1 );

	private readonly prevPosNode = uniform( new Vector3() );

	private readonly prevRightNode = uniform( new Vector3( 1, 0, 0 ) );

	private readonly prevUpNode = uniform( new Vector3( 0, 1, 0 ) );

	private readonly prevFwdNode = uniform( new Vector3( 0, 0, - 1 ) );

	private readonly prevTanYNode = uniform( 1 );

	private readonly prevAspectNode = uniform( 1 );

	private readonly coverageNode = uniform( 0.49 );

	private readonly windOffsetNode = uniform( new Vector2() );

	private readonly windDirNode = uniform( new Vector2( WIND.x, WIND.y ) );

	private readonly evolutionNode = uniform( 0 );

	private readonly windDeltaNode = uniform( new Vector2() );

	private readonly sunNode = uniform( new Vector3( 0.3, 0.6, - 0.7 ).normalize() );

	private readonly nightNode = uniform( 0 );

	private readonly skyIrrNode = uniform( new Vector3( 0.3, 0.4, 0.6 ) );

	private readonly viewHeightNode = uniform( 6360 );

	private readonly pixelConeNode = uniform( 0.001 );

	private readonly stepConeNode = uniform( 0.001 );

	private readonly frameNode = uniform( 0 );

	private readonly historyWeightNode = uniform( HISTORY_WEIGHT );

	private readonly evolveNode = uniform( 0 );

	private readonly historyValidNode = uniform( 0 );

	private readonly validNode = uniform( 0 );

	private readonly tanNode = uniform( new Vector2( 1, 1 ) );

	private weatherKernel: ComputeNode | null = null;

	private boundsKernel: ComputeNode | null = null;

	private traceKernel: ComputeNode | null = null;

	private resolveKernel: ComputeNode | null = null;

	private copyKernel: ComputeNode | null = null;

	private pose: Pose | null = null;

	private prevSun = new Vector3();

	private readonly sky: StorageTexture;

	private readonly transmittance: StorageTexture;

	private disposed = false;

	/**
	 * @param sky 天空视图表
	 * @param transmittance 透射表
	 */
	constructor( sky: StorageTexture, transmittance: StorageTexture ) {

		this.sky = sky;
		this.transmittance = transmittance;
		this.viewClouds = makeHalf( 4, 4, 'skyproHistory' );
		this.historyMeta = makeHalf( 4, 4, 'skyproHistoryMeta' );
		this.viewCamera = {
			right: this.rightNode as Node< 'vec3' >,
			up: this.upNode as Node< 'vec3' >,
			fwd: this.fwdNode as Node< 'vec3' >,
			tan: this.tanNode as Node< 'vec2' >,
			valid: this.validNode as Node< 'float' >,
		};
		void this.loadNoise();

	}

	/**
	 * 推进一帧。噪声还没读完时什么都不画。
	 *
	 * @param renderer 派发 compute 的渲染器
	 * @param coverage 云量，`0..1`。预设是 `0.49`
	 * @param camera 当前相机
	 * @param delta 距上一帧的秒数
	 * @param sun 世界空间太阳方向
	 * @param night 夜深，`0..1`
	 * @param skyIrradiance 天空辐照 `E/π`
	 * @param viewHeightKm 观察点到地心的距离，千米
	 * @param windSpeed 云的平移速度，米/秒
	 * @param evolutionSpeed 形状沿风向的滑动速度，米/秒
	 */
	update( renderer: WebGPURenderer, coverage: number, camera: Camera, delta: number, sun: Vector3, night: number, skyIrradiance: Color, viewHeightKm: number, windSpeed: number, evolutionSpeed: number ): void {

		if ( this.disposed || ! this.noiseReady || ! this.noise || ! this.blue ) return;

		if ( ! this.built ) {

			this.build();
			this.built = true;

		}

		const dt = Math.min( 0.1, Math.max( 0, delta ) );
		const persp = camera as PerspectiveCamera;
		const tanY = Math.tan( ( persp.fov * Math.PI / 180 ) * 0.5 ) / ( persp.zoom || 1 );
		const aspect = persp.aspect || 1;

		renderer.getDrawingBufferSize( this.drawing );

		const viewH = Math.max( 1, this.drawing.y );
		const viewW = Math.max( 1, Math.round( viewH * aspect ) );

		if ( viewW !== this.viewWidth || viewH !== this.viewHeight ) {

			this.viewWidth = viewW;
			this.viewHeight = viewH;
			this.sourceWidth = Math.max( 1, Math.ceil( viewW / 8 ) );
			this.sourceHeight = Math.max( 1, Math.ceil( viewH / 8 ) );
			this.historyWidth = this.sourceWidth * 4;
			this.historyHeight = this.sourceHeight * 4;
			this.source.setSize( this.sourceWidth, this.sourceHeight, 1 );
			this.sourceMeta.setSize( this.sourceWidth, this.sourceHeight, 1 );
			this.scratch.setSize( this.historyWidth, this.historyHeight, 1 );
			this.scratchMeta.setSize( this.historyWidth, this.historyHeight, 1 );
			this.viewClouds.setSize( this.historyWidth, this.historyHeight, 1 );
			this.historyMeta.setSize( this.historyWidth, this.historyHeight, 1 );
			this.historyValid = false;

		}

		if ( coverage !== this.coverageNode.value ) {

			this.coverageNode.value = coverage;
			this.historyValid = false;

		}

		const elements = camera.matrixWorld.elements;
		const right = new Vector3( elements[ 0 ], elements[ 1 ], elements[ 2 ] ).normalize();
		const up = new Vector3( elements[ 4 ], elements[ 5 ], elements[ 6 ] ).normalize();
		const forward = new Vector3( - elements[ 8 ], - elements[ 9 ], - elements[ 10 ] ).normalize();
		const current: Pose = {
			position: camera.position.clone(),
			right,
			up,
			forward,
			tanY,
			aspect,
		};
		const previous = this.pose;

		if ( previous && (
			current.position.distanceTo( previous.position ) > 1000
			|| forward.dot( previous.forward ) < 0.7
			|| Math.abs( tanY - previous.tanY ) > 0.01
			|| sun.dot( this.prevSun ) < 0.999
		) ) this.historyValid = false;

		const from = this.historyValid && previous ? previous : current;

		this.originNode.value.copy( current.position );
		this.rightNode.value.copy( current.right );
		this.upNode.value.copy( current.up );
		this.fwdNode.value.copy( current.forward );
		this.tanYNode.value = tanY;
		this.aspectNode.value = aspect;
		this.tanNode.value.set( tanY * aspect, tanY );
		this.prevPosNode.value.copy( from.position );
		this.prevRightNode.value.copy( from.right );
		this.prevUpNode.value.copy( from.up );
		this.prevFwdNode.value.copy( from.forward );
		this.prevTanYNode.value = from.tanY;
		this.prevAspectNode.value = from.aspect;
		this.sunNode.value.copy( sun );
		this.nightNode.value = night;
		this.skyIrrNode.value.set( skyIrradiance.r, skyIrradiance.g, skyIrradiance.b );
		this.viewHeightNode.value = viewHeightKm;
		this.pixelConeNode.value = 2 * tanY / viewH;
		this.stepConeNode.value = 2 * tanY / this.historyHeight;
		this.frameNode.value = this.frameIndex % 4096;
		this.evolveNode.value = Math.min( 1, evolutionSpeed * dt / 100 );
		this.historyValidNode.value = this.historyValid ? 1 : 0;
		this.historySizeNode.value.set( this.historyWidth, this.historyHeight );
		this.sourceSizeNode.value.set( this.sourceWidth, this.sourceHeight );

		const dx = WIND.x * windSpeed * dt;
		const dz = WIND.y * windSpeed * dt;

		this.windX += dx;
		this.windZ += dz;
		this.evolution += evolutionSpeed * dt;
		this.windOffsetNode.value.set( this.windX, this.windZ );
		this.windDeltaNode.value.set( dx, dz );
		this.evolutionNode.value = this.evolution;

		const [ slotX, slotY ] = this.lattice( this.frameIndex % 16 );

		this.slotNode.value.set( slotX, slotY );

		if ( ! this.weatherDone && this.weatherKernel && this.boundsKernel ) {

			renderer.compute( this.weatherKernel );
			renderer.compute( this.boundsKernel );
			this.weatherDone = true;

		}

		if ( this.traceKernel && this.resolveKernel && this.copyKernel ) {

			this.traceKernel.dispatchSize = [ Math.ceil( this.sourceWidth / 8 ), Math.ceil( this.sourceHeight / 8 ), 1 ];
			this.resolveKernel.dispatchSize = [ Math.ceil( this.historyWidth / 8 ), Math.ceil( this.historyHeight / 8 ), 1 ];
			this.copyKernel.dispatchSize = [ Math.ceil( this.historyWidth / 8 ), Math.ceil( this.historyHeight / 8 ), 1 ];
			renderer.compute( this.traceKernel );
			renderer.compute( this.resolveKernel );
			renderer.compute( this.copyKernel );
			this.validNode.value = 1;
			this.historyValid = true;

		}

		this.pose = current;
		this.prevSun.copy( sun );
		this.frameIndex ++;

	}

	/**
	 * 释放纹理和 compute。形状噪声跟着这一份一起丢掉。
	 *
	 * @param _renderer 未使用。纹理自己释放
	 */
	dispose( _renderer: WebGPURenderer | null ): void {

		this.disposed = true;
		this.weatherKernel?.dispose();
		this.boundsKernel?.dispose();
		this.traceKernel?.dispose();
		this.resolveKernel?.dispose();
		this.copyKernel?.dispose();
		this.weather.dispose();
		this.bounds.dispose();
		this.source.dispose();
		this.sourceMeta.dispose();
		this.scratch.dispose();
		this.scratchMeta.dispose();
		this.viewClouds.dispose();
		this.historyMeta.dispose();
		this.noise?.dispose();
		this.blue?.dispose();

	}

	private lattice( index: number ): [ number, number ] {

		let x = 0;
		let y = 0;

		for ( let bit = 0; ( 1 << bit ) < 4; bit ++ ) {

			const pair = ( index >> ( bit * 2 ) ) & 3;

			x |= ( ( pair >> 1 ) ^ ( pair & 1 ) ) << ( 2 - bit - 1 );
			y |= ( pair & 1 ) << ( 2 - bit - 1 );

		}

		return [ x, y ];

	}

	private async loadNoise(): Promise< void > {

		try {

			const [ shapeResponse, blueResponse ] = await Promise.all( [
				fetch( '/clouds/baseShape64.bin' ),
				fetch( '/clouds/blueNoise.bin' ),
			] );
			const packed = new Uint8Array( await new Response( shapeResponse.body!.pipeThrough( new DecompressionStream( 'gzip' ) ) ).arrayBuffer() );
			const header = new DataView( packed.buffer, packed.byteOffset, packed.byteLength );
			const width = header.getUint16( 6, true );
			const height = header.getUint16( 8, true );
			const depth = header.getUint16( 10, true );
			const pixels = packed.slice( 16, 16 + width * height * depth * 4 );
			const noise = new DataTexture( packNoiseAtlas( pixels, width, height, depth ), width * 8, height * 8 );

			noise.name = 'skyproShape';
			noise.format = RGBAFormat;
			noise.type = UnsignedByteType;
			noise.colorSpace = NoColorSpace;
			noise.wrapS = ClampToEdgeWrapping;
			noise.wrapT = ClampToEdgeWrapping;
			noise.magFilter = NearestFilter;
			noise.minFilter = NearestFilter;
			noise.generateMipmaps = false;
			noise.flipY = false;
			noise.needsUpdate = true;

			const blue = new DataTexture( new Uint8Array( await blueResponse.arrayBuffer() ), 64, 64, RedFormat, UnsignedByteType );

			blue.name = 'skyproBlue';
			blue.colorSpace = NoColorSpace;
			blue.wrapS = RepeatWrapping;
			blue.wrapT = RepeatWrapping;
			blue.magFilter = LinearFilter;
			blue.minFilter = LinearFilter;
			blue.generateMipmaps = false;
			blue.needsUpdate = true;

			if ( this.disposed ) {

				noise.dispose();
				blue.dispose();
				return;

			}

			this.noise = noise;
			this.blue = blue;
			this.noiseReady = true;

		} catch ( error ) {

			console.warn( '体积云噪声没有读入', error );

		}

	}

	private build(): void {

		const noise = this.noise;
		const blue = this.blue;

		if ( ! noise || ! blue ) return;

		const weatherKernel = Fn( () => {

			const uv = vec2( float( globalId.x ), float( globalId.y ) ).div( 1024 );
			const value = skyproWeather( { uv } ) as Node< 'vec4' >;

			textureStore( this.weather, globalId.xy, value ).toWriteOnly();

		} )().computeKernel( [ 8, 8, 1 ] );

		weatherKernel.dispatchSize = [ 1024 / 8, 1024 / 8, 1 ];
		weatherKernel.setName( 'SkyPro.Weather' );
		this.weatherKernel = weatherKernel as ComputeNode;

		const boundsKernel = Fn( () => {

			const id = uvec2( globalId.x, globalId.y );
			const maximum = skyproBounds( { id, weather: storageTexture( this.weather ).toReadOnly() } ) as Node< 'float' >;

			textureStore( this.bounds, globalId.xy, vec4( maximum, 0, 0, 1 ) ).toWriteOnly();

		} )().computeKernel( [ 8, 8, 1 ] );

		boundsKernel.dispatchSize = [ 8, 8, 1 ];
		boundsKernel.setName( 'SkyPro.Bounds' );
		this.boundsKernel = boundsKernel as ComputeNode;

		const noiseNode = texture( noise );
		const weatherNode = texture( this.weather );
		const traceKernel = Fn( () => {

			const px = vec2( float( globalId.x ), float( globalId.y ) );
			const inside = px.x.lessThan( this.sourceSizeNode.x ).and( px.y.lessThan( this.sourceSizeNode.y ) );

			If( inside, () => {

				const uv = px.mul( 4 ).add( this.slotNode ).add( 0.5 ).div( this.historySizeNode );
				const blueSample = textureLoad( blue, uvec2( globalId.x.mod( 64 ), globalId.y.mod( 64 ) ) ) as Node< 'vec4' >;
				const dither = fract( blueSample.x.add( this.frameNode.mul( 0.61803398875 ) ) );
				const depth = float( 0 ).toVar();
				const marched = skyproTrace( {
					uv,
					dither,
					pixelCone: this.pixelConeNode,
					stepCone: this.stepConeNode,
					origin: this.originNode,
					right: this.rightNode,
					up: this.upNode,
					forward: this.fwdNode,
					tanY: this.tanYNode,
					aspect: this.aspectNode,
					coverage: this.coverageNode,
					windOffset: this.windOffsetNode,
					windDir: this.windDirNode,
					evolution: this.evolutionNode,
					sunDir: this.sunNode,
					night: this.nightNode,
					skyIrr: this.skyIrrNode,
					viewHeight: this.viewHeightNode,
					depthOut: depth,
					noise: noiseNode,
					weather: weatherNode,
					weatherSmp: sampler( texture( this.weather ) ),
					bounds: storageTexture( this.bounds ).toReadOnly(),
					sky: storageTexture( this.sky ).toReadOnly(),
					trans: storageTexture( this.transmittance ).toReadOnly(),
				} ) as Node< 'vec4' >;

				textureStore( this.source, globalId.xy, marched ).toWriteOnly();
				textureStore( this.sourceMeta, globalId.xy, vec4( depth.div( 1000 ), 1, 0, 0 ) ).toWriteOnly();

			} );

		} )().computeKernel( [ 8, 8, 1 ] );

		traceKernel.setName( 'SkyPro.Trace' );
		this.traceKernel = traceKernel as ComputeNode;

		const resolveKernel = Fn( () => {

			const px = vec2( float( globalId.x ), float( globalId.y ) );
			const inside = px.x.lessThan( this.historySizeNode.x ).and( px.y.lessThan( this.historySizeNode.y ) );

			If( inside, () => {

				const meta = vec4( 0, 0, 0, 0 ).toVar();
				const color = skyproResolve( {
					id: uvec2( globalId.x, globalId.y ),
					historySize: this.historySizeNode,
					sourceSize: this.sourceSizeNode,
					slot: this.slotNode,
					origin: this.originNode,
					right: this.rightNode,
					up: this.upNode,
					forward: this.fwdNode,
					tanY: this.tanYNode,
					aspect: this.aspectNode,
					prevPos: this.prevPosNode,
					prevRight: this.prevRightNode,
					prevUp: this.prevUpNode,
					prevFwd: this.prevFwdNode,
					prevTanY: this.prevTanYNode,
					prevAspect: this.prevAspectNode,
					windDelta: this.windDeltaNode,
					historyWeight: this.historyWeightNode,
					evolve: this.evolveNode,
					historyValid: this.historyValidNode,
					outMeta: meta,
					curColor: storageTexture( this.source ).toReadOnly(),
					curMeta: storageTexture( this.sourceMeta ).toReadOnly(),
					prevColor: texture( this.viewClouds ),
					prevSmp: sampler( texture( this.viewClouds ) ),
					prevMeta: storageTexture( this.historyMeta ).toReadOnly(),
				} ) as Node< 'vec4' >;

				textureStore( this.scratch, globalId.xy, color ).toWriteOnly();
				textureStore( this.scratchMeta, globalId.xy, meta ).toWriteOnly();

			} );

		} )().computeKernel( [ 8, 8, 1 ] );

		resolveKernel.setName( 'SkyPro.Resolve' );
		this.resolveKernel = resolveKernel as ComputeNode;

		const copyKernel = Fn( () => {

			const px = vec2( float( globalId.x ), float( globalId.y ) );

			If( px.x.lessThan( this.historySizeNode.x ).and( px.y.lessThan( this.historySizeNode.y ) ), () => {

				const color = textureLoad( this.scratch, uvec2( globalId.x, globalId.y ) ) as Node< 'vec4' >;
				const meta = textureLoad( this.scratchMeta, uvec2( globalId.x, globalId.y ) ) as Node< 'vec4' >;

				textureStore( this.viewClouds, globalId.xy, color ).toWriteOnly();
				textureStore( this.historyMeta, globalId.xy, meta ).toWriteOnly();

			} );

		} )().computeKernel( [ 8, 8, 1 ] );

		copyKernel.setName( 'SkyPro.Copy' );
		this.copyKernel = copyKernel as ComputeNode;

	}

}
