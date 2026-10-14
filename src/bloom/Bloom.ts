/**
 * 泛光。Jimenez 2014 那套：半分辨率开始五级缩小，再逐级叠回来。
 *
 * 第一级用 Karis 平均，避免太阳这种单个亮点闪烁。加回画面时乘 `bloom`，默认 `0.05`。
 * `meter` 是缩到 `1/16` 的那一级，自动曝光读它。
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

const SCALES = [ 0.5, 0.25, 0.125, 0.0625, 0.03125 ];

let quad: QuadMesh | null = null;
const size = new Vector2();
type RendererState = Parameters<typeof RendererUtils.restoreRendererState>[ 1 ];

let saved: RendererState = {} as RendererState;

function target( name: string ): RenderTarget {

	const rt = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
	rt.texture.name = name;
	rt.texture.generateMipmaps = false;
	return rt;

}

function material( fragment: Node ): NodeMaterial {

	const result = new NodeMaterial();
	result.fragmentNode = fragment;
	result.toneMapped = false;
	result.depthTest = false;
	result.depthWrite = false;
	return result;

}

function tap( map: TextureNode, texel: Node< 'vec2' >, x: number, y: number ): Node< 'vec3' > {

	const sampled = map.sample( uv().add( texel.mul( vec2( x, y ) ) ) ) as Node< 'vec4' >;

	return sampled.rgb;

}

function karis( color: Node< 'vec3' > ): Node< 'vec3' > {

	return color.div( luminance( color ).add( 1 ) ) as Node< 'vec3' >;

}

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

			const g0 = karis( j.add( k ).add( l ).add( m ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.5 );
			const g1 = karis( a.add( b ).add( d ).add( e ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const g2 = karis( b.add( c ).add( e ).add( f ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const g3 = karis( d.add( e ).add( g ).add( h ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const g4 = karis( e.add( f ).add( h ).add( i ).mul( 0.25 ) as Node< 'vec3' > ).mul( 0.125 );
			const sum = g0.add( g1 ).add( g2 ).add( g3 ).add( g4 ) as Node< 'vec3' >;

			return vec4( sum.div( max( float( 1 ).sub( luminance( sum ) ), 0.02 ) ), 1 );

		}

		const sum = e.mul( 0.125 )
			.add( a.add( c ).add( g ).add( i ).mul( 0.03125 ) )
			.add( b.add( d ).add( f ).add( h ).mul( 0.0625 ) )
			.add( j.add( k ).add( l ).add( m ).mul( 0.125 ) );

		return vec4( sum, 1 );

	} )();

}

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
 * 场景泛光。接进输出节点后，每帧在画最终画面之前自己渲染。
 */
export class SceneBloom extends TempNode {

	/** 缩到 1/16 的一级，给自动曝光计量。 */
	readonly meter: Texture;

	/** 加回画面的倍数。默认 `0.05`。 */
	readonly bloom = uniform( 0.05 );

	private readonly down: RenderTarget[];

	private readonly up: RenderTarget[];

	private readonly downFirst: NodeMaterial;

	private readonly downNext: NodeMaterial;

	private readonly downSource: TextureNode;

	private readonly upPass: NodeMaterial;

	private readonly upSmall: TextureNode;

	private readonly upBase: TextureNode;

	/**
	 * @param input 场景颜色
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

	setup(): Node {

		return texture( this.up[ 0 ].texture ).mul( this.bloom );

	}

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

		const small = [ this.down[ 4 ], this.up[ 3 ], this.up[ 2 ], this.up[ 1 ] ];
		const base = [ this.down[ 3 ], this.down[ 2 ], this.down[ 1 ], this.down[ 0 ] ];

		for ( let index = 0; index < this.up.length; index ++ ) {

			this.upSmall.value = small[ index ].texture;
			this.upBase.value = base[ index ].texture;
			this.draw( renderer, this.upPass, this.up[ 3 - index ] );

		}

		RendererUtils.restoreRendererState( renderer, saved );

	}

	dispose(): void {

		super.dispose();

		for ( const rt of this.down ) rt.dispose();
		for ( const rt of this.up ) rt.dispose();
		this.downFirst.dispose();
		this.downNext.dispose();
		this.upPass.dispose();

	}

	private draw( renderer: WebGPURenderer, pass: NodeMaterial, rt: RenderTarget ): void {

		if ( ! quad ) quad = new QuadMesh( pass );
		renderer.setRenderTarget( rt );
		quad.material = pass;
		quad.render( renderer );

	}

}
