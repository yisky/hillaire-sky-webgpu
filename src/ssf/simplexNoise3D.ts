import { Fn, abs, dot, float, floor, max, min, mod, step, vec3, vec4 } from 'three/tsl'

// 排列表。同一整数恒定，相邻整数不相关。289 是 Simplex 常用的模，接近 7×7×6
const permute = /*@__PURE__*/ Fn( ( [ x ]: [ any ] ) => mod( x.mul( 34 ).add( 1 ).mul( x ), 289 ) )

/**
 * 3D Simplex 噪声。
 *
 * 每个立方体切成 6 个四面体，只采样 4 个角点。输出已从 [ -1, 1 ] 映射到 [ 0, 1 ]，和 @tx3d/core 的 simplexNoise3D 一致。
 * 雾效会再映射回 [ -1, 1 ]，用来抬高或压低雾面。
 */
export const simplexNoise3D = /*@__PURE__*/ Fn( ( [ v ]: [ any ] ) => {

	// unskew = 1/6，skew = 1/3。把立方体斜切成四面体，第四个角点偏移 0.5
	const unskew = float( 1 / 6 )
	const skew = float( 1 / 3 )

	const pos = vec3( v )
	const i = floor( pos.add( dot( pos, vec3( skew ) ) ) )
	const x0: any = pos.sub( i ).add( dot( i, vec3( unskew ) ) )

	// 与 yzx 旋转比较，得到当前四面体的 i1 / i2 角点偏移
	const gt: any = step( vec3( x0.y, x0.z, x0.x ), vec3( x0.x, x0.y, x0.z ) )
	const lt: any = vec3( 1 ).sub( gt )
	const i1: any = min( gt, vec3( lt.z, lt.x, lt.y ) )
	const i2: any = max( gt, vec3( lt.z, lt.x, lt.y ) )

	const x1 = x0.sub( i1 ).add( unskew )
	const x2 = x0.sub( i2 ).add( skew )
	const x3 = x0.sub( 0.5 )

	// 四个角点分别置换，得到互不相关的梯度索引
	const ii: any = mod( i, 289 )
	const perm: any = permute(
		permute(
			permute( ii.z.add( vec4( 0, i1.z, i2.z, 1 ) ) )
				.add( ii.y )
				.add( vec4( 0, i1.y, i2.y, 1 ) )
		)
			.add( ii.x )
			.add( vec4( 0, i1.x, i2.x, 1 ) )
	)

	// 7×7 格点映射到八面体梯度。289 接近 49×6
	const nScale = float( 1 / 7 )
	const ns = vec3( nScale.mul( 2 ), nScale.mul( 0.5 ), nScale ).sub( vec3( 0, 1, 0 ) )

	const j = perm.sub( float( 49 ).mul( floor( perm.mul( ns.z ).mul( ns.z ) ) ) )
	const jx = floor( j.mul( ns.z ) )
	const jy = floor( j.sub( jx.mul( 7 ) ) )

	const ox: any = jx.mul( ns.x ).add( ns.y )
	const oy: any = jy.mul( ns.x ).add( ns.y )
	const h = vec4( 1 ).sub( abs( ox ) ).sub( abs( oy ) )

	const b0 = vec4( ox.x, ox.y, oy.x, oy.y )
	const b1 = vec4( ox.z, ox.w, oy.z, oy.w )

	const s0 = floor( b0 ).mul( 2 ).add( 1 )
	const s1 = floor( b1 ).mul( 2 ).add( 1 )
	const sh = vec4( 0 ).sub( step( h, vec4( 0 ) ) )

	const a0 = vec4( b0.x, b0.z, b0.y, b0.w ).add(
		vec4( s0.x, s0.z, s0.y, s0.w ).mul( vec4( sh.x, sh.x, sh.y, sh.y ) )
	)
	const a1 = vec4( b1.x, b1.z, b1.y, b1.w ).add(
		vec4( s1.x, s1.z, s1.y, s1.w ).mul( vec4( sh.z, sh.z, sh.w, sh.w ) )
	)

	const grad0 = vec3( a0.x, a0.y, h.x )
	const grad1 = vec3( a0.z, a0.w, h.y )
	const grad2 = vec3( a1.x, a1.y, h.z )
	const grad3 = vec3( a1.z, a1.w, h.w )

	// taylorInvSqrt 近似归一化四个梯度，避免对每个角点做 sqrt
	const lenSq = vec4( dot( grad0, grad0 ), dot( grad1, grad1 ), dot( grad2, grad2 ), dot( grad3, grad3 ) )
	const norm = float( 1.79284291400159 ).sub( float( 0.85373472095314 ).mul( lenSq ) )

	const n0 = dot( grad0.mul( norm.x ), x0 )
	const n1 = dot( grad1.mul( norm.y ), x1 )
	const n2 = dot( grad2.mul( norm.z ), x2 )
	const n3 = dot( grad3.mul( norm.w ), x3 )

	// 径向衰减。0.6 是四面体里角点贡献的半径，四次方让远离角点的贡献迅速归零
	const m = max( float( 0.6 ).sub( vec4( dot( x0, x0 ), dot( x1, x1 ), dot( x2, x2 ), dot( x3, x3 ) ) ), 0 )
	const m2 = m.mul( m )

	// 42 把结果缩放到约 [ -1, 1 ]，再平移到 [ 0, 1 ]
	return float( 42 ).mul( dot( m2.mul( m2 ), vec4( n0, n1, n2, n3 ) ) ).mul( 0.5 ).add( 0.5 )

}, { v: 'vec3', return: 'float' } )
