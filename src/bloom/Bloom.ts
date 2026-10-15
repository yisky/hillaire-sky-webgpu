/**
 * 泛光。Jimenez 2014 那套：半分辨率开始五级缩小，再逐级叠回来。
 *
 * 第一级用 Karis 平均，避免太阳这种单个亮点闪烁。加回画面时乘 `bloom`，默认 `0.05`。
 * `meter` 是缩到 `1/16` 的那一级，自动曝光读它，不读已经加过炫光的结果。
 *
 * 接法走 {@link TempNode}：图里写成 `color.add( bloom )`，`setup` 只返回泛光贡献，不含场景色。
 * `updateBefore` 在画最终画面之前自己渲金字塔。没有 compute。
 */

import {
	HalfFloatType,
	NodeMaterial,
	NodeUpdateType,
	QuadMesh,
	RenderTarget,
	TempNode,
	RendererUtils,
	Vector2,
	type Node,
	type NodeFrame,
	type Texture,
	type TextureNode,
	type WebGPURenderer,
} from 'three/webgpu';
import { Fn, float, int, luminance, max, texture, textureSize, uniform, uv, vec2, vec4 } from 'three/tsl';

/** 五级缩小：1/2、1/4、1/8、1/16、1/32。放大四级叠回半分辨率。 */
const SCALES = [ 0.5, 0.25, 0.125, 0.0625, 0.03125 ];

/** 各级共用一张全屏四边形，换材质再画。 */
let quad: QuadMesh | null = null;
const size = new Vector2();
type RendererState = Parameters<typeof RendererUtils.restoreRendererState>[ 1 ];

/** 画金字塔前把渲染器状态存下来，画完还原，免得弄脏最终画面那趟。 */
let saved: RendererState = {} as RendererState;

/** 半浮点、不写深度、不生成 mip。 */
function target( name: string ): RenderTarget {

	const rt = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
	rt.texture.name = name;
	rt.texture.generateMipmaps = false;
	return rt;

}

/** 全屏后期材质。不做色调映射，也不测深度。 */
function material( fragment: Node ): NodeMaterial {

	const result = new NodeMaterial();
	result.fragmentNode = fragment;
	result.toneMapped = false;
	result.depthTest = false;
	result.depthWrite = false;
	return result;

}

/**
 * 相对当前像素偏 `(x, y)` 个纹素采样。
 *
 * @param map 源图
 * @param texel 一个纹素的 UV 尺寸
 * @param x 水平偏移，单位纹素
 * @param y 竖直偏移，单位纹素
 */
function tap( map: TextureNode, texel: Node< 'vec2' >, x: number, y: number ): Node< 'vec3' > {

	const sampled = map.sample( uv().add( texel.mul( vec2( x, y ) ) ) ) as Node< 'vec4' >;

	return sampled.rgb;

}

/** Karis 平均：颜色除以亮度加一，亮点权变小，太阳盘不会闪一像素。 */
function karis( color: Node< 'vec3' > ): Node< 'vec3' > {

	return color.div( luminance( color ).add( 1 ) ) as Node< 'vec3' >;

}

/**
 * 缩小一级。13 个点：十字加四角，再加内圈四角。
 * 第一级先按 2×2 块做 Karis，再除一次亮度，压掉高光。后面几级只做普通加权。
 *
 * @param map 上一级
 * @param first 是不是从场景色缩到 1/2
 */
function downsample( map: TextureNode, first: boolean ): Node {

	return Fn( () => {

		const texel = vec2( 1, 1 ).div( textureSize( map, int( 0 ) ) as unknown as Node< 'vec2' > ) as Node< 'vec2' >;
		const a = tap( map, texel, - 2, - 2 );
		const b = tap( map, texel, 0, - 2 );
		const c = tap( map, texel, 2, - 2 );
		const d = tap( map, texel, - 2, 0 );
		const e = tap( map, texel, 0, 0 );
		const f = tap( map, texel, 2, 0 );
		const g = tap( map, texel, - 2, 2 );
		const h = tap( map, texel, 0, 2 );
		const i = tap( map, texel, 2, 2 );
		const j = tap( map, texel, - 1, - 1 );
		const k = tap( map, texel, 1, - 1 );
		const l = tap( map, texel, - 1, 1 );
		const m = tap( map, texel, 1, 1 );

		if ( first ) {

			// 内圈四角权 1/2，外圈四块各 1/8。再除 `max(1 - lum, 0.02)`，特别亮的块几乎加不进去。
			const g0 = karis( j.add( k ).add( l ).add( m ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.5 );
			const g1 = karis( a.add( b ).add( d ).add( e ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const g2 = karis( b.add( c ).add( e ).add( f ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const g3 = karis( d.add( e ).add( g ).add( h ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const g4 = karis( e.add( f ).add( h ).add( i ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const sum = g0.add( g1 ).add( g2 ).add( g3 ).add( g4 ) as Node< 'vec3' >;

			return vec4( sum.div( max( float( 1 ).sub( luminance( sum ) ), 0.02 ) ), 1 );

		}

		// 中心 1/8，十字 1/16，四角 1/32，内圈四角各 1/8。
		const sum = e.mul( 0.125 )
			.add( a.add( c ).add( g ).add( i ).mul( 0.03125 ) )
			.add( b.add( d ).add( f ).add( h ).mul( 0.0625 ) )
			.add( j.add( k ).add( l ).add( m ).mul( 0.125 ) );

		return vec4( sum, 1 );

	} )();

}

/**
 * 放大一级， tent 3×3（中心 4、十字 2、四角 1，除 16）叠到这一级的缩小结果上。
 *
 * @param small 更糊的那张
 * @param base 当前这一级的缩小图
 */
function upsample( small: TextureNode, base: TextureNode ): Node {

	return Fn( () => {

		const texel = vec2( 1, 1 ).div( textureSize( small, int( 0 ) ) as unknown as Node< 'vec2' > ) as Node< 'vec2' >;
		const center = tap( small, texel, 0, 0 ).mul( 4 );
		const cross = tap( small, texel, - 1, 0 ).add( tap( small, texel, 1, 0 ) ).add( tap( small, texel, 0, - 1 ) ).add( tap( small, texel, 0, 1 ) ).mul( 2 );
		const corners = tap( small, texel, - 1, - 1 ).add( tap( small, texel, 1, - 1 ) ).add( tap( small, texel, - 1, 1 ) ).add( tap( small, texel, 1, 1 ) );
		const sum = center.add( cross ).add( corners ).div( 16 );

		const baseColor = base.sample( uv() ) as Node< 'vec4' >;

		return vec4( baseColor.rgb.add( sum ), 1 );

	} )();

}

/**
 * 场景泛光。
 *
 * 构造时传入场景颜色。雾的输入写成 `color.add( bloom )`。
 * 每帧 `pipeline.render()` 时自己渲金字塔，没有单独的 compute。
 */
export class SceneBloom extends TempNode {

	/** 缩到 1/16 的一级，给自动曝光计量。 */
	readonly meter: Texture;

	/** 加回画面的倍数。默认 `0.05`。运行时改 `.value`。 */
	readonly bloom = uniform( 0.05 );

	/** 五级缩小。`down[3]` 就是 `meter`。 */
	private readonly down: RenderTarget[];

	/** 四级放大。`up[0]` 是半分辨率的最终泛光。 */
	private readonly up: RenderTarget[];

	/** 从场景色缩到 1/2，走 Karis。 */
	private readonly downFirst: NodeMaterial;

	/** 后面几级缩小，普通加权。源图是 `downSource`。 */
	private readonly downNext: NodeMaterial;

	/** 下一级缩小要读的图。每级画之前改 `.value`。 */
	private readonly downSource: TextureNode;

	/** 放大一级。小图 `upSmall`，当前级 `upBase`。 */
	private readonly upPass: NodeMaterial;

	/** 放大时更糊的那张。 */
	private readonly upSmall: TextureNode;

	/** 放大时叠回去的当前级。 */
	private readonly upBase: TextureNode;

	/**
	 * @param input 场景通道的颜色
	 */
	constructor( input: TextureNode ) {

		super( 'vec4' );

		this.updateBeforeType = NodeUpdateType.FRAME;
		this.down = SCALES.map( ( _, index ) => target( 'bloomDown' + index ) );
		this.up = SCALES.slice( 0, 4 ).map( ( _, index ) => target( 'bloomUp' + index ) );
		this.meter = this.down[ 3 ].texture;

		this.downFirst = material( downsample( input, true ) );
		this.downSource = texture( this.down[ 0 ].texture );
		this.downNext = material( downsample( this.downSource, false ) );
		this.upSmall = texture( this.down[ 4 ].texture );
		this.upBase = texture( this.down[ 3 ].texture );
		this.upPass = material( upsample( this.upSmall, this.upBase ) );

	}

	/**
	 * 半分辨率泛光乘倍数。接在场景色后面、雾前面，用加法叠回去。
	 */
	setup(): Node {

		return texture( this.up[ 0 ].texture ).mul( this.bloom );

	}

	/**
	 * 按绘制缓冲大小重设各级，再缩小五级、放大四级。
	 * 画之前把渲染器状态存下来，画完还原。
	 *
	 * @param frame 当前节点帧
	 */
	updateBefore( frame: NodeFrame ): boolean | undefined {

		const renderer = frame.renderer as WebGPURenderer | null;

		if ( ! renderer ) return;

		saved = RendererUtils.resetRendererState( renderer, saved );
		renderer.getDrawingBufferSize( size );

		SCALES.forEach( ( scale, index ) => {

			this.down[ index ].setSize( Math.max( 1, Math.round( size.x * scale ) ), Math.max( 1, Math.round( size.y * scale ) ) );

		} );
		SCALES.slice( 0, 4 ).forEach( ( scale, index ) => {

			this.up[ index ].setSize( Math.max( 1, Math.round( size.x * scale ) ), Math.max( 1, Math.round( size.y * scale ) ) );

		} );

		this.draw( renderer, this.downFirst, this.down[ 0 ] );

		for ( let index = 1; index < this.down.length; index ++ ) {

			this.downSource.value = this.down[ index - 1 ].texture;
			this.draw( renderer, this.downNext, this.down[ index ] );

		}

		// 从最糊往回叠：1/32 叠到 1/16，再叠到 1/8、1/4，最后到 1/2。
		const small = [ this.down[ 4 ], this.up[ 3 ], this.up[ 2 ], this.up[ 1 ] ];
		const base = [ this.down[ 3 ], this.down[ 2 ], this.down[ 1 ], this.down[ 0 ] ];

		for ( let index = 0; index < this.up.length; index ++ ) {

			this.upSmall.value = small[ index ].texture;
			this.upBase.value = base[ index ].texture;
			this.draw( renderer, this.upPass, this.up[ 3 - index ] );

		}

		RendererUtils.restoreRendererState( renderer, saved );

		return;

	}

	/** 金字塔和三份材质。共用的四边形不在这里丢，其它后期还可能用。 */
	dispose(): void {

		super.dispose();

		for ( const rt of this.down ) rt.dispose();
		for ( const rt of this.up ) rt.dispose();
		this.downFirst.dispose();
		this.downNext.dispose();
		this.upPass.dispose();

	}

	/**
	 * 用共用四边形把 `pass` 画进 `rt`。
	 *
	 * @param renderer 当前渲染器
	 * @param pass 这一级的材质
	 * @param rt 这一级的目标
	 */
	private draw( renderer: WebGPURenderer, pass: NodeMaterial, rt: RenderTarget ): void {

		if ( ! quad ) quad = new QuadMesh( pass );
		renderer.setRenderTarget( rt );
		quad.material = pass;
		quad.render( renderer );

	}

}
