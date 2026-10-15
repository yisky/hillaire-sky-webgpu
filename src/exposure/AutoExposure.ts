/**
 * 自动曝光。单独的后期组件，不读天空模块。
 *
 * 整幅画面乘同一个倍数，把中心加权的对数平均亮度慢慢拉向 `ref`。
 * 偏暗时按 `up` 加快，偏亮时按 `down` 减慢。倍数限制在 `min` 到 `max`。
 * `enabled` 小于等于 `0.5` 时倍数保持 `1`。这几项运行时改 `.value`。
 * 太阳低于地平线时上限收到 `2`，夜空不会被拉成白天。阈值和天空里恒星出现的区间一致：
 * `-sun.y` 从 `0.02` 到 `0.18`。
 *
 * 接法跟泛光一样走 {@link TempNode}：图里直接用这个节点，`setup` 返回乘过倍数的颜色。
 * `updateBefore` 只写 `dt`。适应不能放在 `updateAfter` 里派发——那时画布那趟还没提交，
 * compute 会排到最终画面前面，倍数变成当帧生效。所以仍由外面在 `pipeline.render()` 之后
 * `renderer.compute(adapt)`，倍数晚一帧。
 * 计量读泛光缩到 `1/16` 的那张图，一个 256 线程的组把每个纹素都加一遍。
 */

import {
	StorageBufferAttribute,
	TempNode,
	Vector3,
	NodeUpdateType,
	type ComputeNode,
	type Node,
	type NodeFrame,
	type Texture,
} from 'three/webgpu';
import {
	Fn,
	If,
	Loop,
	exp,
	exp2,
	float,
	int,
	ivec2,
	length,
	localId,
	log2,
	luminance,
	max,
	mix,
	pow,
	smoothstep,
	storage,
	texture,
	textureLoad,
	textureSize,
	uint,
	vec2,
	vec4,
	uniform,
	workgroupArray,
	workgroupBarrier,
} from 'three/tsl';

/** 一个工作组的线程数。计量图按这个宽度拆开加。 */
const GROUP = 256;
/** 每个线程最多扫这么多纹素。256×160 盖住 1/16 计量图。 */
const SPAN = 160;
/** 夜里倍数上限。太阳降到地平线下后从 `max` 收到这里。 */
const NIGHT_MAX = 2;

export interface AutoExposureOptions {

	/** 场景通道的颜色。计量读它，不读已经加过炫光的结果。 */
	color: Texture;

	/** 已经叠过炫光的场景颜色。`setup` 用当前倍数缩放它。 */
	input: Node< 'vec4' >;

}

/**
 * 自动曝光。
 *
 * 构造时传入计量图和要缩放的颜色。调色的输入就是这个节点。
 * 每帧先抄好 {@link AutoExposure.sunDirection}，再 `pipeline.render()`，最后
 * `renderer.compute(exposure.adapt)`。
 */
export class AutoExposure extends TempNode {

	/** 适应。场景画完后派发，下一帧才用新的倍数。 */
	readonly adapt: ComputeNode;

	/** 大于 `0.5` 才适应，否则倍数保持 `1`。 */
	readonly enabled = uniform( 1 );

	/** 不需要修正的平均亮度。着色器里叫 `refLum`，因为 `ref` 是 WGSL 关键字。 */
	readonly ref = uniform( 0.25 );

	/** 倍数下限。 */
	readonly min = uniform( 0.6 );

	/** 倍数上限。夜里另收到 `2`。 */
	readonly max = uniform( 6 );

	/** 变亮的速度，单位 1/秒。 */
	readonly up = uniform( 1.6 );

	/** 变暗的速度，单位 1/秒。 */
	readonly down = uniform( 1.1 );

	/** 指向太阳的世界方向。每帧写 `.value`。夜里用 `-y` 收曝光上限。 */
	readonly sunDirection = uniform( new Vector3( 0, 1, 0 ) );

	/** 距上一帧的秒数，钳在 `0..0.1`。只给适应用，乘法不用它。 */
	private readonly dt = uniform( 1 / 60 );

	/** 当前倍数。片段着色器只读，适应计算可写。初值 `1`。 */
	private readonly factor: Node< 'float' >;

	/** 已经叠过炫光的颜色。 */
	private readonly input: Node< 'vec4' >;

	/**
	 * @param options 计量图和要缩放的颜色
	 */
	constructor( options: AutoExposureOptions ) {

		super( 'vec4' );

		this.updateBeforeType = NodeUpdateType.FRAME;
		this.input = options.input;
		// 同一块存储：着色器只读 `factor`，适应计算写另一头。画完再 compute，避免当帧改倍数。

		const attribute = new StorageBufferAttribute( new Float32Array( [ 1 ] ), 1 );
		const written = storage( attribute, 'float', 1 );
		const read = storage( attribute, 'float', 1 ).toReadOnly();

		this.factor = read.element( 0 ) as Node< 'float' >;
		this.adapt = this.createAdapt( written.element( 0 ) as Node< 'float' >, options.color );

	}

	/**
	 * 用上一帧算好的倍数缩放颜色。接在炫光后面、调色前面。
	 */
	setup(): Node {

		return vec4( this.input.rgb.mul( this.factor ), this.input.a );

	}

	/**
	 * 写下这一帧的时间间隔。适应计算用它，乘法不用。
	 * 不要在这里 `compute`：计量图要等泛光的 `updateBefore` 画完，而且 compute 必须排在整趟 render 提交之后。
	 *
	 * @param frame 当前节点帧
	 */
	updateBefore( frame: NodeFrame ): boolean | undefined {

		this.dt.value = Math.min( Math.max( frame.deltaTime, 0 ), 0.1 );

		return;

	}

	/**
	 * 量这一帧的亮度，并写上下一帧要用的倍数。
	 * 中心权重大、四角小。目标亮度偏高时把比值开 0.8 次方，避免瞬间拉爆。
	 *
	 * @param current 可写的倍数，初值是 `1`
	 * @param color 泛光缩到 1/16 的计量图
	 */
	private createAdapt( current: Node< 'float' >, color: Texture ): ComputeNode {

		const image = texture( color );
		const dt = this.dt;
		const enabled = this.enabled;
		const ref = this.ref;
		const min = this.min;
		const exposureMax = this.max;
		const up = this.up;
		const down = this.down;
		const sumL = workgroupArray( 'float', GROUP ) as unknown as { element( index: Node ): Node< 'float' > };
		const sumW = workgroupArray( 'float', GROUP ) as unknown as { element( index: Node ): Node< 'float' > };

		const built = Fn( () => {

			const thread = localId.x;
			const size = textureSize( image, int( 0 ) ) as unknown as Node< 'uvec2' >;
			const count = size.x.mul( size.y );
			const accL = float( 0 ).toVar();
			const accW = float( 0 ).toVar();

			// 256 个线程各扫一条，步长 GROUP，把整张计量图走完。
			Loop( { start: uint( 0 ), end: uint( SPAN ), type: 'uint' }, ( { i } ) => {

				const index = thread.add( i.mul( uint( GROUP ) ) );

				If( index.lessThan( count ), () => {

					const x = index.mod( size.x );
					const y = index.div( size.x );
					const sample = textureLoad( image, ivec2( int( x ), int( y ) ), int( 0 ) ) as unknown as Node< 'vec4' >;
					const centered = vec2( float( x ).add( 0.5 ).div( float( size.x ) ), float( y ).add( 0.5 ).div( float( size.y ) ) ).sub( 0.5 );
					// 中心权重大，竖直方向收得更紧，最暗也留 0.15。
					const weight = max( float( 1 ).sub( length( centered.mul( vec2( 1, 1.4 ) ) ).mul( 1.2 ) ), 0.15 );
					const logLum = log2( max( luminance( sample.rgb ), 1e-4 ) );

					accL.addAssign( logLum.mul( weight ) );
					accW.addAssign( weight );

				} );

			} );

			sumL.element( thread ).assign( accL );
			sumW.element( thread ).assign( accW );
			workgroupBarrier();
			// 组内折半相加，0 号线程拿到整张图的加权对数和。

			for ( let span = GROUP / 2; span > 0; span >>= 1 ) {

				const step = uint( span );

				If( thread.lessThan( step ), () => {

					sumL.element( thread ).addAssign( sumL.element( thread.add( step ) ) );
					sumW.element( thread ).addAssign( sumW.element( thread.add( step ) ) );

				} );

				workgroupBarrier();

			}

			If( thread.equal( uint( 0 ) ), () => {

				const average = exp2( sumL.element( uint( 0 ) ).div( max( sumW.element( uint( 0 ) ), 1e-4 ) ) );
				const ratio = ref.div( average );
				const partial = ratio.greaterThan( 1 ).select( pow( ratio, 0.8 ), ratio );
				// 太阳落到地平线下后收上限，夜空不会被拉成白天。
				const night = smoothstep( 0.02, 0.18, this.sunDirection.y.negate() );
				const cap = mix( exposureMax, NIGHT_MAX, night );
				const target = partial.clamp( min, cap );
				const rate = target.greaterThan( current ).select( up, down );
				const step = float( 1 ).sub( exp( dt.mul( rate ).negate() ) );
				// 对数空间插值，避免线性混合把倍数拉过零。
				const next = exp2( mix( log2( max( current, 1e-3 ) ), log2( target ), step ) );

				current.assign( enabled.greaterThan( 0.5 ).select( next, float( 1 ) ) );

			} );

		} )();

		return ( built as unknown as { compute( count: number[], size: number[] ): ComputeNode } ).compute( [ 1, 1, 1 ], [ GROUP, 1, 1 ] );

	}

}
