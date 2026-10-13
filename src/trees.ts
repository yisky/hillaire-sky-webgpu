import {
	BufferGeometry,
	Group,
	InstancedMesh,
	Mesh,
	Object3D,
	Quaternion,
	Vector3,
	type Camera,
	type Material,
} from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// tree0 116 面放 14850 棵，tree1 11630 面放 2000 棵。
const QUOTA = [ 14850, 2000 ];
const AREA = [ 110, 110 ];
const ORIGIN = [ 0, - 0.09, 0 ];
const SCALE = [ 0.7, 1.3 ];
const AVOID = 4;
const SEED = 22;
const KINDS = 2;
// 主楼旁三棵树约 0.72、0.79、0.93 米。林子以 0.8 米为中心，再乘 0.7 到 1.3，高度错开。
const TARGET_HEIGHT = 0.8;
const MODELS = [ '/models/tree0.glb', '/models/tree1.glb' ];

// 分布图覆盖的地面范围，中心相对原点有一点偏移。
const MAP_SIZE = 189;
const MAP_OFFSET = [ 4.5, - 2.2 ];

type Tree = {
	x: number;
	y: number;
	z: number;
	scale: number;
	width: number;
	kind: number;
	yaw: number;
};

function hash( text: string ): number {

	let value = 0;

	for ( let i = 0; i < text.length; i ++ ) {

		value = ( ( value << 5 ) - value + text.charCodeAt( i ) ) | 0;

	}

	return value >>> 0;

}

function random( seed: number ): () => number {

	let state = hash( String( seed ) );

	return () => {

		state = ( state + 1831565813 ) | 0;
		let value = state;
		value = Math.imul( value ^ ( value >>> 15 ), value | 1 );
		value ^= value + Math.imul( value ^ ( value >>> 7 ), value | 61 );
		return ( ( value ^ ( value >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

function loadImage( url: string ): Promise< HTMLImageElement > {

	return new Promise( ( resolve, reject ) => {

		const image = new Image();
		image.onload = () => resolve( image );
		image.onerror = reject;
		image.src = url;

	} );

}

function readPixels( image: HTMLImageElement ): Uint8ClampedArray {

	const canvas = document.createElement( 'canvas' );
	canvas.width = image.width;
	canvas.height = image.height;
	const context = canvas.getContext( '2d' );
	if ( ! context ) throw new Error( '无法读取分布图。' );
	context.drawImage( image, 0, 0 );
	return context.getImageData( 0, 0, image.width, image.height ).data;

}

function sampleMap( pixels: Uint8ClampedArray, width: number, height: number, x: number, z: number ): number {

	const u = Math.min( 1, Math.max( 0, ( x - MAP_OFFSET[ 0 ] ) / MAP_SIZE + 0.5 ) );
	const v = Math.min( 1, Math.max( 0, ( z - MAP_OFFSET[ 1 ] ) / MAP_SIZE + 0.5 ) );
	const column = Math.floor( u * ( width - 1 ) );
	const row = Math.floor( v * ( height - 1 ) );
	return pixels[ ( row * width + column ) * 4 ] / 255;

}

function place( pixels: Uint8ClampedArray, width: number, height: number ): Tree[] {

	const next = random( SEED );
	const trees: Tree[] = [];
	const got = [ 0, 0 ];
	const limit = ( QUOTA[ 0 ] + QUOTA[ 1 ] ) * 40;
	let tries = 0;

	while ( got[ 0 ] < QUOTA[ 0 ] || got[ 1 ] < QUOTA[ 1 ] ) {

		if ( tries >= limit ) break;
		tries ++;
		const x = ORIGIN[ 0 ] + ( next() - 0.5 ) * AREA[ 0 ];
		const z = ORIGIN[ 2 ] + ( next() - 0.5 ) * AREA[ 1 ];
		if ( Math.hypot( x - ORIGIN[ 0 ], z - ORIGIN[ 2 ] ) < AVOID ) continue;
		const density = sampleMap( pixels, width, height, x, z );
		// 源码是阈值 0.01，再掷一次 density * 10。少掷这一次，后面每棵的坐标都会错开。
		if ( density < 0.01 ) continue;
		if ( next() > density * 10 ) continue;
		const scaleRoll = next();
		const kindRoll = next();
		const yawRoll = next();
		const kind = Math.floor( kindRoll * KINDS );
		if ( got[ kind ] >= QUOTA[ kind ] ) continue;
		got[ kind ] ++;
		trees.push( {
			x,
			y: ORIGIN[ 1 ],
			z,
			scale: SCALE[ 0 ] + scaleRoll * ( SCALE[ 1 ] - SCALE[ 0 ] ),
			width: 1,
			kind,
			yaw: yawRoll * Math.PI * 2,
		} );

	}

	return trees;

}

function cutout( material: Material ) {

	const leaf = material as Material & { transparent: boolean; alphaTest: number; depthWrite: boolean; needsUpdate: boolean };
	if ( ! leaf.transparent ) return;
	leaf.transparent = false;
	leaf.alphaTest = 0.1;
	leaf.depthWrite = true;
	leaf.needsUpdate = true;

}

function attributeKey( geometry: BufferGeometry ): string {

	return Object.keys( geometry.attributes ).sort().join( ',' );

}

// 树干在模型原点附近，树冠会偏向一侧。只把树干底落到 y=0，不按整棵包围盒去对中，否则树干会离开采样点。
function trunkBase( geometries: BufferGeometry[], span: number, minY: number ): number {

	const radius = span * 0.12;
	const radiusSq = radius * radius;
	let base = Infinity;

	for ( const geometry of geometries ) {

		const position = geometry.getAttribute( 'position' );

		for ( let i = 0; i < position.count; i ++ ) {

			const x = position.getX( i );
			const z = position.getZ( i );
			if ( x * x + z * z > radiusSq ) continue;
			base = Math.min( base, position.getY( i ) );

		}

	}

	return Number.isFinite( base ) ? base : minY;

}

function fit( meshes: Mesh[] ): Mesh[] {

	const baked: { geometry: BufferGeometry; material: Material }[] = [];

	for ( const mesh of meshes ) {

		const geometry = mesh.geometry.clone();
		geometry.applyMatrix4( mesh.matrixWorld );
		mesh.geometry.dispose();
		const material = mesh.material;
		if ( Array.isArray( material ) ) continue;
		baked.push( { geometry, material } );

	}

	let minX = Infinity;
	let maxX = - Infinity;
	let minY = Infinity;
	let maxY = - Infinity;
	let minZ = Infinity;
	let maxZ = - Infinity;

	for ( const item of baked ) {

		item.geometry.computeBoundingBox();
		const box = item.geometry.boundingBox;
		if ( ! box ) continue;
		minX = Math.min( minX, box.min.x );
		maxX = Math.max( maxX, box.max.x );
		minY = Math.min( minY, box.min.y );
		maxY = Math.max( maxY, box.max.y );
		minZ = Math.min( minZ, box.min.z );
		maxZ = Math.max( maxZ, box.max.z );

	}

	const height = maxY - minY;
	const scale = height > 0 ? TARGET_HEIGHT / height : 1;
	const span = Math.max( maxX - minX, maxZ - minZ );
	const ground = trunkBase( baked.map( ( item ) => item.geometry ), span, minY ) * scale;

	for ( const item of baked ) {

		item.geometry.scale( scale, scale, scale );
		item.geometry.translate( 0, - ground, 0 );

	}

	const groups = new Map<Material, BufferGeometry[]>();

	for ( const item of baked ) {

		const list = groups.get( item.material ) ?? [];
		list.push( item.geometry );
		groups.set( item.material, list );

	}

	const result: Mesh[] = [];

	for ( const [ material, geometries ] of groups ) {

		cutout( material );
		const key = attributeKey( geometries[ 0 ] );
		const merged = geometries.length > 1 && geometries.every( ( geometry ) => attributeKey( geometry ) === key )
			? mergeGeometries( geometries, false )
			: null;

		if ( merged ) {

			for ( const geometry of geometries ) geometry.dispose();
			result.push( new Mesh( merged, material ) );

		} else {

			for ( const geometry of geometries ) result.push( new Mesh( geometry, material ) );

		}

	}

	return result;

}

function loadModel( url: string ): Promise<Mesh[]> {

	const loader = new GLTFLoader();

	return new Promise( ( resolve, reject ) => {

		loader.load( url, ( gltf ) => {

			gltf.scene.updateMatrixWorld( true );
			const meshes: Mesh[] = [];
			gltf.scene.traverse( ( obj ) => {

				const mesh = obj as Mesh;
				if ( mesh.isMesh ) meshes.push( mesh );

			} );
			resolve( fit( meshes ) );

		}, undefined, reject );

	} );

}

function loadSpecies(): Promise<Mesh[][]> {

	return Promise.all( MODELS.map( loadModel ) );

}

function fill( parts: Mesh[], trees: Tree[], castShadow: boolean ): InstancedMesh[] {

	const dummy = new Object3D();
	const position = new Vector3();
	const rotation = new Quaternion();
	const scale = new Vector3();
	const meshes: InstancedMesh[] = [];

	for ( const part of parts ) {

		const instanced = new InstancedMesh( part.geometry, part.material, trees.length );
		instanced.frustumCulled = false;
		instanced.castShadow = castShadow;
		instanced.receiveShadow = true;

		for ( let i = 0; i < trees.length; i ++ ) {

			const tree = trees[ i ];
			position.set( tree.x, tree.y, tree.z );
			rotation.setFromAxisAngle( Object3D.DEFAULT_UP, tree.yaw );
			scale.set( tree.scale * tree.width, tree.scale, tree.scale * tree.width );
			dummy.matrix.compose( position, rotation, scale );
			instanced.setMatrixAt( i, dummy.matrix );

		}

		meshes.push( instanced );

	}

	return meshes;

}

function scatter( parts: Mesh[], trees: Tree[], castRadius: number ): InstancedMesh[] {

	const near: Tree[] = [];
	const far: Tree[] = [];
	const radiusSq = castRadius * castRadius;

	for ( const tree of trees ) {

		if ( tree.x * tree.x + tree.z * tree.z <= radiusSq ) near.push( tree );
		else far.push( tree );

	}

	return [
		...( near.length ? fill( parts, near, true ) : [] ),
		...( far.length ? fill( parts, far, false ) : [] ),
	];

}

function disposeMaterial( material: Material ) {

	const maps = material as Material & {
		map?: { dispose(): void } | null;
		normalMap?: { dispose(): void } | null;
		roughnessMap?: { dispose(): void } | null;
		metalnessMap?: { dispose(): void } | null;
		aoMap?: { dispose(): void } | null;
	};

	maps.map?.dispose();
	maps.normalMap?.dispose();
	maps.roughnessMap?.dispose();
	maps.metalnessMap?.dispose();
	maps.aoMap?.dispose();
	material.dispose();

}

export type Forest = {
	object: Group;
	update( camera: Camera ): void;
	dispose(): void;
};

export function createForest( castRadius: number ): Promise< Forest > {

	const map = loadImage( '/trees/distribution.webp' ).then( ( image ) => {

		const pixels = readPixels( image );
		return place( pixels, image.width, image.height );

	} );

	return Promise.all( [ map, loadSpecies() ] ).then( ( [ trees, species ] ) => {

		const group = new Group();

		for ( let kind = 0; kind < species.length; kind ++ ) {

			const placed = trees.filter( ( tree ) => tree.kind === kind );
			for ( const mesh of scatter( species[ kind ], placed, castRadius ) ) group.add( mesh );

		}

		return {
			object: group,
			update() {},
			dispose() {

				const geometries = new Set<BufferGeometry>();
				const materials = new Set<Material>();

				group.traverse( ( obj ) => {

					const mesh = obj as InstancedMesh;
					if ( ! mesh.isInstancedMesh ) return;
					geometries.add( mesh.geometry );
					const list = Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ];
					for ( const material of list ) materials.add( material );

				} );

				for ( const geometry of geometries ) geometry.dispose();
				for ( const material of materials ) disposeMaterial( material );

			},
		};

	} );

}
