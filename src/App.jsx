import { useEffect, useRef } from 'react'
import * as THREE from 'three/webgpu'
import { builtinAOContext, float, Fn, mix, pass, screenUV, texture, uniform, vec4 } from 'three/tsl'
import { ao } from 'three/addons/tsl/display/GTAONode.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js'
import { SunLight } from 'three/addons/lights/SunLight.js'
import { SunLightNode } from 'three/addons/lights/SunLightNode.js'
import { Sky } from './sky'
import { SkyProField } from './sky/clouds/skypro'
import { Glare } from './glare'
import { AutoExposure } from './exposure'
import { createForest, cutout } from './trees'
import { SceneBloom } from './bloom/Bloom'
import { Grade } from './grade/Grade'
import { Display } from './grade/Display'
import { SSF } from './ssf/SSF'
import { SkyProbe } from './env/SkyProbe'
import { createFps } from './ui/fps'
import { createSettings } from './ui/settings'

/**
 * 演示入口。全屏 canvas 上画天空、场地、树林和后期。
 *
 * 轨道相机默认在西南看向原点（主楼视觉中心，不是场地组的位移）。
 * 第一人称按 1.8 米视高，场地整体缩放 0.08，所以眼睛高度是 `1.8 * 0.08`。
 * `?demo=1` 跑 120 秒时间轴，结束后把控制交回轨道。
 *
 * 后期不走渲染器自带的色调映射。输出节点是 Display：ACES、sRGB、抖动。
 * 炫光可见度和自动曝光适应在 `pipeline.render()` 之后 compute，晚一帧生效。
 */
const ORBIT_HOME = { position: [ - 5, 1.2, 5 ], target: [ 0, 1, 0 ], fov: 45 }
/** 第一人称眼睛离地高度。1.8 米乘场地缩放。 */
const EYE = 1.8 * 0.08
/** 步行速度，米/秒，不乘场地缩放。Shift 用 WALK_FAST。 */
const WALK = 1.4
const WALK_FAST = 3
/** 第一人称最低离地，防止掉到模型下面。 */
const EYE_MIN = 0.04
/** 轨道近裁。第一人称更近，否则近处栏杆会被切掉。远裁都是 1000。 */
const ORBIT_NEAR = 0.1
const FPS_NEAR = 0.02
/** 轨道俯仰下限，避免翻到相机正下方。上限在创建时设成 π/2，不钻地。 */
const ORBIT_MIN_POLAR = 0.05
const DOWN = new THREE.Vector3( 0, - 1, 0 )
/** `?demo=1` 走时间轴。结束后 `demoDone`，轨道和面板可以手动用。 */
const DEMO = new URLSearchParams( location.search ).has( 'demo' )
const DEMO_END = 120
/** 西南 3/4 方位，主楼在画面左中，不被右侧面板挡住。 */
const Q = - 0.75
/** `Q` 再加一圈。时间轴里用它避免方位从 2π 跳回 0 时倒转。 */
const Q2 = Q + Math.PI * 2
/**
 * 演示关键帧。当前全是轨道机位。
 * `t` 秒，`hour` 太阳时刻；相邻帧方位、高度、半径、目标用 smoothstep，云量用线性 `u`。
 * `up` 抬头看云。日落钉在 17.8、日出钉在 6.2（太阳贴地平线，18.3 已经沉下去）。
 */
const DEMO_KEYS = [
	{ t: 0, hour: 10, mode: 'orbit', az: - 0.79, el: 0.03, r: 7.1, cov: 0.49 },
	{ t: 5, hour: 10, mode: 'orbit', az: - 0.4, el: 0.03, r: 7.1, cov: 0.49 },
	{ t: 6, hour: 10, mode: 'orbit', az: - 0.4, el: 0.03, r: 7.1, cov: 0.49, up: 1, panel: 1 },
	{ t: 8, hour: 10, mode: 'orbit', az: - 0.4, el: 0.03, r: 7.1, cov: 0.2, up: 1, panel: 1 },
	{ t: 11, hour: 10, mode: 'orbit', az: - 0.4, el: 0.03, r: 7.1, cov: 0.7, up: 1, panel: 1 },
	{ t: 14, hour: 10, mode: 'orbit', az: - 0.4, el: 0.03, r: 7.1, cov: 0.45, up: 1, panel: 1 },
	{ t: 18, hour: 10, mode: 'orbit', az: Q, el: 0.42, r: 11, cov: 0.45, panel: 1 },
	{ t: 22, hour: 10, mode: 'orbit', az: Q + 0.12, el: 0.4, r: 10.6, cov: 0.45, panel: 1 },
	{ t: 27, hour: 10, mode: 'orbit', az: 0.85, el: 0.32, r: 10, cov: 0.45, panel: 1 },
	{ t: 32, hour: 10, mode: 'orbit', az: 0.85 + Math.PI, el: 0.25, r: 10, cov: 0.45, panel: 1 },
	{ t: 34, hour: 10, mode: 'orbit', az: 0.85 + Math.PI, el: 0.18, r: 5.5, cov: 0.45, panel: 1 },
	{ t: 44, hour: 10, mode: 'orbit', az: 0.85 + Math.PI + 2.62, el: 0.18, r: 5.5, cov: 0.45, panel: 1 },
	{ t: 47, hour: 11, mode: 'orbit', az: Q2, el: 0.42, r: 11, cov: 0.45, panel: 1 },
	{ t: 61, hour: 17.8, mode: 'orbit', az: Q2, el: 0.42, r: 11, cov: 0.45, panel: 1 },
	{ t: 64, hour: 17.8, mode: 'orbit', az: Q2 + 0.1, el: 0.38, r: 10.8, cov: 0.45, panel: 1 },
	{ t: 71, hour: 17.8, mode: 'orbit', az: Math.PI / 2 + Math.PI * 2, el: 0.12, r: 11, ty: 1.2, tz: - 1.5, cov: 0.45, panel: 1 },
	{ t: 76, hour: 17.8, mode: 'orbit', az: Math.PI / 2 + Math.PI * 2 + 0.12, el: 0.1, r: 10.5, ty: 1.15, tz: - 1.4, cov: 0.45, panel: 1 },
	{ t: 79, hour: 22, mode: 'orbit', az: Q2, el: 0.42, r: 11, cov: 0.45, panel: 1 },
	{ t: 87, hour: 22, mode: 'orbit', az: Q2 + 0.45, el: 0.42, r: 11, cov: 0.45, panel: 1 },
	{ t: 93, hour: 22, mode: 'orbit', az: Q2 + 0.55, el: 0.38, r: 10.5, cov: 0.45, up: 0.35, panel: 1 },
	{ t: 96, hour: 6.2, mode: 'orbit', az: Math.PI * 1.5, el: 0.1, r: 8, ty: 1.2, tz: 1.5, cov: 0.45, panel: 1 },
	{ t: 110, hour: 6.4, mode: 'orbit', az: Math.PI * 1.5 + 0.45, el: 0.1, r: 7, ty: 1.2, tz: 1.5, cov: 0.45, panel: 1 },
	{ t: 120, hour: 7.6, mode: 'orbit', az: Q2, el: 0.42, r: 11, cov: 0.45, panel: 1 },
]
/** 钟点按 24 小时环插值，不会从 23 跳到 6 时倒着转。 */
const lerpHour = ( a, b, t ) => ( a + ( ( b - a + 24 ) % 24 ) * t ) % 24

// 离原点超过这个距离的物体不投射阴影。日月方向或相机超出下面的幅度才重画阴影。
const SHADOW_CAST_RADIUS = 40
const SHADOW_DIR = 0.01
const SHADOW_MOVE = 1
const SHADOW_TURN = 0.02

// 整组模型先缩到 0.08，再挪到这个位置。道路在组内再抬高一点。
const SITE_FILES = [
	[ '/models/building-transformed.glb', 0 ],
	[ '/models/building_floor-transformed.glb', 0 ],
	[ '/models/building_trees-transformed.glb', 0 ],
	[ '/models/landscape-transformed.glb', 0 ],
	[ '/models/terrain.glb', 0 ],
	[ '/models/roads-transformed.glb', 0.019 ],
]

const SUN_UP = new THREE.Vector3( 0, 1, 0 )

// 太阳位置：北纬 24°、赤纬 6°。返回指向太阳的方向，+x 东、−z 北、+y 上。
function sunDirectionFromTime( hours, out ) {

	const phi = THREE.MathUtils.degToRad( 24 )
	const dec = THREE.MathUtils.degToRad( 6 )
	const hourAngle = THREE.MathUtils.degToRad( ( hours - 12 ) * 15 )
	const east = - Math.cos( dec ) * Math.sin( hourAngle )
	const north = Math.cos( phi ) * Math.sin( dec ) - Math.sin( phi ) * Math.cos( dec ) * Math.cos( hourAngle )
	const up = Math.sin( phi ) * Math.sin( dec ) + Math.cos( phi ) * Math.cos( dec ) * Math.cos( hourAngle )
	return out.set( east, up, - north ).normalize()

}

/** 把钟点换成天空的太阳方向，再绕世界 Y 转方位。 */
function applySun( sky, hours, azimuth ) {

	sunDirectionFromTime( hours, sky.sunDirection ).applyAxisAngle( SUN_UP, THREE.MathUtils.degToRad( azimuth ) )

}

/** 正午雾密度。贴地平线更浓。 */
const FOG_DENSITY_NOON = 0.005
/** 日出前后的雾密度。 */
const FOG_DENSITY_HORIZON = 0.012
/** 辐射雾开始散掉的时刻。 */
const FOG_DAWN = 5.8
/** 辐射雾散尽、回到正午密度的时刻。 */
const FOG_CLEAR = 10
/** 后半夜开始积雾。 */
const FOG_GATHER = 3.5
/** 雾体参考色。`matchFogLuma` 只改色相，亮度钉在这层。 */
const FOG_REFERENCE = new THREE.Color( '#d0dce8' )
const FOG_LUMA = 0.2126 * FOG_REFERENCE.r + 0.7152 * FOG_REFERENCE.g + 0.0722 * FOG_REFERENCE.b
/** 逆光散射相对主光的倍数。 */
const FOG_INSCATTER = 1.5
/** 白天太阳照度参考。逆光雾除以它，太阳落下后强度跟着掉。 */
const SUN_ILLUMINANCE = 11

// 辐射雾：日出前后最浓，上午散掉。日落到后半夜保持稀，黎明前几小时才积起来，夜里才看得见天空。
function radiationFogDensity( hours ) {

	const h = ( ( hours % 24 ) + 24 ) % 24
	if ( h >= FOG_DAWN && h < FOG_CLEAR ) {

		return THREE.MathUtils.lerp( FOG_DENSITY_HORIZON, FOG_DENSITY_NOON, THREE.MathUtils.smoothstep( h, FOG_DAWN, FOG_CLEAR ) )

	}
	if ( h >= FOG_GATHER && h < FOG_DAWN ) {

		return THREE.MathUtils.lerp( FOG_DENSITY_NOON, FOG_DENSITY_HORIZON, THREE.MathUtils.smoothstep( h, FOG_GATHER, FOG_DAWN ) )

	}
	return FOG_DENSITY_NOON

}

/** Rec.709 亮度。给雾色钉亮度用，不进着色器。 */
function colorLuma( color ) {

	return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b

}

// 只取色相，亮度钉在现在这层雾上。源色必须先是散射光，不能是日盘透射。
function matchFogLuma( out, source ) {

	const luma = colorLuma( source )
	out.copy( source )
	if ( luma > 1e-4 ) out.multiplyScalar( FOG_LUMA / luma )
	else out.copy( FOG_REFERENCE )
	return out

}

/** 月亮放在太阳水平方向的对面，高度压低但仍在天上。只改方向，颜色和亮度由天空写。 */
function placeMoon( sky ) {

	const sun = sky.sunDirection

	sky.moonDirection.set( - sun.x, Math.abs( sun.y ) * 0.8 + 0.25, - sun.z ).normalize()

}

/** 卸几何和材质。贴图跟 GLTF 走，这里不单独丢。 */
function disposeObject( root ) {

	root.traverse( ( obj ) => {

		obj.geometry?.dispose()
		const materials = obj.material
		if ( ! materials ) return
		for ( const material of Array.isArray( materials ) ? materials : [ materials ] ) material.dispose()

	} )

}

/**
 * 全屏画布。挂载时建场景、加载模型和后期，卸载时整条链拆掉。
 * 没有 WebGPU 时只在画布上写一句提示。
 */
export default function App() {

	const canvasRef = useRef( null )
	const noteRef = useRef( null )

	useEffect( () => {

		const canvas = canvasRef.current
		const settings = createSettings()
		const camera = new THREE.PerspectiveCamera( ORBIT_HOME.fov, 1, ORBIT_NEAR, 1000 )
		camera.position.set( ...ORBIT_HOME.position )
		camera.lookAt( ...ORBIT_HOME.target )
		const timer = new THREE.Timer()
		timer.connect( document )
		const scene = new THREE.Scene()
		const sunDir = new THREE.Vector3()
		sunDirectionFromTime( settings.sun.time, sunDir ).applyAxisAngle( SUN_UP, THREE.MathUtils.degToRad( settings.sun.azimuth ) )
		const sky = new Sky( { sunDirection: sunDir } )
		// 体积云读天空查找表。主视图行进结果给合成和炫光遮挡。
		const clouds = new SkyProField( sky.skyView, sky.transmittance )
		sky.clouds = clouds
		sky.coverage = settings.coverage
		scene.add( sky )

		const sunLight = new SunLight( 0xffffff, 1 )
		const moonLight = new SunLight( 0xffffff, 0 )
		// 环境贴图已经提供天空的漫反射。这里强度为 0，每帧只抄 skyIrradiance 当颜色，楼不会亮两遍。
		const ambient = new THREE.AmbientLight( 0xffffff, 0 )
		// 阴影：2048，近 30、远 80，范围 ±25。bias -0.0005，normalBias 0.05。
		// 两盏灯都开投射，用 shadow.intensity 只让更亮的那盏出影，避免各渲一遍。贴图不自动每帧重画。
		sunLight.castShadow = true
		sunLight.shadow.intensity = 1.0;
		moonLight.castShadow = true
		moonLight.shadow.intensity = 0.0;
		for ( const light of [ sunLight, moonLight ] ) {

			light.shadow.autoUpdate = false
			light.shadow.mapSize.set( 2048, 2048 )
			light.shadow.camera.near = 30
			light.shadow.camera.far = 80
			light.shadow.camera.left = -25
			light.shadow.camera.right = 25
			light.shadow.camera.top = 25
			light.shadow.camera.bottom = -25
			light.shadow.bias = -0.0005
			light.shadow.normalBias = 0.05

		}
		scene.add( sunLight, moonLight, ambient )

		// 场地缩到 0.08 再挪开。树林不进这个组，跟场地平级，用世界坐标。
		const site = new THREE.Group()
		site.scale.setScalar( 0.08 )
		site.position.set( 5.8, 0, 1 )
		scene.add( site )

		const draco = new DRACOLoader()
		draco.setDecoderPath( '/draco/' )
		const loader = new GLTFLoader()
		loader.setDRACOLoader( draco )

		// 轨道和第一人称共用一台相机。切换时尽量保住 xz，第一人称 y 落到地面加视高。
		const orbit = new OrbitControls( camera, canvas )
		orbit.target.set( ...ORBIT_HOME.target )
		orbit.enableDamping = false
		orbit.zoomSpeed = 0.5
		orbit.minDistance = 4
		orbit.maxDistance = 25
		orbit.minPolarAngle = ORBIT_MIN_POLAR
		orbit.maxPolarAngle = Math.PI / 2
		orbit.enabled = ! DEMO
		orbit.update()
		const look = new PointerLockControls( camera, canvas )
		const lookDir = new THREE.Vector3()
		const ray = new THREE.Raycaster()
		const rayFrom = new THREE.Vector3()
		const euler = new THREE.Euler( 0, 0, 0, 'YXZ' )
		let camMode = 'orbit'
		/** 地形、地面、道路上的落点高度。只打 `userData.walk`，不打楼和树。 */
		const groundY = ( x, z ) => {

			rayFrom.set( x, 20, z )
			ray.set( rayFrom, DOWN )
			const hit = ray.intersectObject( site, true ).find( ( item ) => item.object.userData.walk )
			return hit ? hit.point.y : 0

		}
		const standY = ( x, z ) => groundY( x, z ) + EYE
		/**
		 * 在轨道和第一人称之间切。第一人称点画布锁定指针，Esc 解锁。V 键走设置。
		 * 切到轨道时用当前朝向重算 target，距离钉在 `minDistance`。切到第一人称时抹掉 roll，y 落到地面加视高。
		 */
		const applyCam = ( mode ) => {

			if ( mode !== 'first' && mode !== 'orbit' ) return
			if ( mode === camMode ) return
			if ( look.isLocked ) look.unlock()
			camMode = mode
			orbit.enabled = mode === 'orbit'
			if ( mode === 'orbit' ) {

				camera.near = ORBIT_NEAR
				camera.getWorldDirection( lookDir )
				orbit.target.copy( camera.position ).addScaledVector( lookDir, orbit.minDistance )
				orbit.update()

			} else {

				camera.near = FPS_NEAR
				euler.setFromQuaternion( camera.quaternion, 'YXZ' )
				euler.x = 0
				euler.z = 0
				camera.position.y = standY( camera.position.x, camera.position.z )
				camera.quaternion.setFromEuler( euler )

			}

			camera.updateProjectionMatrix()

		}
		const keys = new Set()
		let demoT = 0
		let demoDone = false
		let demoPanel = false
		const demoTarget = new THREE.Vector3( ...ORBIT_HOME.target )
		const demoFocus = new THREE.Vector3()
		/**
		 * 按 DEMO_KEYS 插值时刻、云量、轨道相机。
		 * 方位切到另一段时用前一帧的 `az` 钉住，避免 2π 跳变倒转。
		 * 演示未结束时关掉手动轨道。
		 */
		const runDemo = ( elapsed ) => {

			const last = DEMO_KEYS[ DEMO_KEYS.length - 1 ]
			let a = last
			let b = last
			let u = 1
			let s = 1
			if ( elapsed < last.t ) {

				let i = 0
				while ( i + 1 < DEMO_KEYS.length && DEMO_KEYS[ i + 1 ].t <= elapsed ) i ++
				a = DEMO_KEYS[ i ]
				b = DEMO_KEYS[ i + 1 ]
				u = ( elapsed - a.t ) / ( b.t - a.t || 1 )
				s = u * u * ( 3 - 2 * u )

			}

			settings.sun.setTime( lerpHour( a.hour, b.hour, s ) )
			if ( a.cov != null ) settings.setCoverage( a.cov + ( ( b.cov ?? a.cov ) - a.cov ) * u )
			if ( ( a.panel || b.panel ) && ! demoPanel ) {

				settings.open( 'sky' )
				demoPanel = true

			}

			settings.setCamera( a.mode )
			applyCam( a.mode )
			const hold = b.mode !== 'orbit'
			const az = a.az + ( ( hold ? a.az : b.az ) - a.az ) * s
			const el = a.el + ( ( hold ? a.el : b.el ) - a.el ) * s
			const radius = a.r + ( ( hold ? a.r : b.r ) - a.r ) * s
			const upK = ( a.up || 0 ) + ( ( ( hold ? a.up : b.up ) || 0 ) - ( a.up || 0 ) ) * s
			const tx = ( a.tx ?? 0 ) + ( ( hold ? a.tx ?? 0 : b.tx ?? 0 ) - ( a.tx ?? 0 ) ) * s
			const ty = ( a.ty ?? 1 ) + ( ( hold ? a.ty ?? 1 : b.ty ?? 1 ) - ( a.ty ?? 1 ) ) * s
			const tz = ( a.tz ?? 0 ) + ( ( hold ? a.tz ?? 0 : b.tz ?? 0 ) - ( a.tz ?? 0 ) ) * s
			const ce = Math.cos( el )
			demoTarget.set( tx, ty, tz )
			camera.position.set(
				demoTarget.x + radius * Math.sin( az ) * ce,
				Math.max( 0.2, demoTarget.y + radius * Math.sin( el ) ),
				demoTarget.z + radius * Math.cos( az ) * ce,
			)
			demoFocus.copy( demoTarget )
			if ( upK > 0 ) {

				lookDir.copy( demoTarget ).sub( camera.position )
				lookDir.y += upK * lookDir.length() * 2.2
				demoFocus.copy( camera.position ).add( lookDir )

			}

			camera.lookAt( demoFocus )

		}
		/** 第一人称点画布才锁指针。演示没跑完不锁。 */
		const onCanvasClick = () => {

			if ( DEMO && ! demoDone ) return
			if ( camMode === 'first' && ! look.isLocked ) look.lock()

		}
		const onKeyDown = ( e ) => {

			if ( e.target && ( e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA' ) ) return
			keys.add( e.code )
			if ( e.code === 'Space' ) e.preventDefault()
			if ( e.code === 'KeyV' && ( ! DEMO || demoDone ) ) {

				e.preventDefault()
				settings.setCamera( camMode === 'orbit' ? 'first' : 'orbit' )

			}

		}
		const onKeyUp = ( e ) => keys.delete( e.code )
		const onBlur = () => keys.clear()
		canvas.addEventListener( 'click', onCanvasClick )
		window.addEventListener( 'keydown', onKeyDown )
		window.addEventListener( 'keyup', onKeyUp )
		window.addEventListener( 'blur', onBlur )

		let renderer = null
		let pipeline = null
		let prePass = null
		let depthMaterial = null
		let depthWhite = null
		let aoPass = null
		let scenePass = null
		let forest = null
		let bloom = null
		let fog = null
		let glare = null
		let exposure = null
		let grade = null
		let display = null
		let environment = null
		let fps = null
		const lightDirection = new THREE.Vector3()
		const fogDay = new THREE.Color()
		const fogNight = new THREE.Color()
		const fogTint = new THREE.Color()
		const fogInscatter = new THREE.Color()
		let stopped = false
		let onResize = null
		// 场地若干件加树林一件。都到齐才开演示和阴影，避免空场景先烤一帧。
		let loadsLeft = SITE_FILES.length + 1
		let contentReady = false
		const markLoaded = () => {

			loadsLeft --
			if ( loadsLeft === 0 ) contentReady = true

		}
		/**
		 * 上一帧画阴影时的光方向和相机。太阳月亮各记一份。
		 * 切回来不必立刻重画；方向、相机位移或转向超出阈值才 `needsUpdate`。
		 */
		const shadowSlot = () => ( {
			dir: new THREE.Vector3(),
			pos: new THREE.Vector3(),
			quat: new THREE.Quaternion(),
			ready: false,
		} )
		const sunShadow = shadowSlot()
		const moonShadow = shadowSlot()
		const bound = new THREE.Sphere()

		// 树林跟场地平级。超出 SHADOW_CAST_RADIUS 的实例不投射阴影。
		createForest( SHADOW_CAST_RADIUS ).then( ( next ) => {

			if ( stopped ) {

				next.dispose()
				markLoaded()
				return

			}

			forest = next
			scene.add( forest.object )
			markLoaded()

		} ).catch( () => {

			markLoaded()

		} )

		for ( const [ url, height ] of SITE_FILES ) {

			loader.load( url, ( gltf ) => {

				if ( stopped ) {

					disposeObject( gltf.scene )
					return

				}

				gltf.scene.position.y = height
				// 地形、地面、道路可踩。楼和树不打 walk，第一人称不会站上去。
				// 先开投射和接收，加进场地算出世界包围球后再把远处的关掉，阴影贴图留给近处。
				const walk = /terrain|landscape|roads/.test( url )
				const foliage = /building_trees/.test( url )
				gltf.scene.traverse( ( obj ) => {

					if ( walk ) obj.userData.walk = true
					if ( obj.isMesh ) {

						obj.castShadow = true
						obj.receiveShadow = true

					}

					const materials = obj.material
					if ( ! materials ) return
					for ( const material of Array.isArray( materials ) ? materials : [ materials ] ) {

						// 主楼旁三棵树枝叶是 BLEND，卡片交叉会排错。改成和林子一样的镂空。
						if ( foliage ) cutout( material )

						// 中心地表：几乎不金属，法线强度反过来压一档，免得鼓包太硬。
						if ( material.name === 'TerrainCenterNew' ) {

							material.metalness = 0.3
							material.roughness = 0.95
							material.normalScale?.set( - 0.25, - 0.5 )

						}

						// 导出时被遮罩丢掉的阳台玻璃：半透明、能反射环境。
						if ( material.name === 'vray balcony glass' ) {

							material.color.set( '#2d424c' )
							material.transparent = true
							material.opacity = 0.51
							material.alphaTest = 0
							material.metalness = 0
							material.roughness = 0
							material.transmission = 0
							material.specularIntensity = 1
							material.clearcoat = 1
							material.ior = 1.4
							material.iridescence = 0.4
							material.iridescenceIOR = 1.28
							material.iridescenceThicknessRange = [ 0.9, 0.2 ]
							material.envMapIntensity = 0.5
							material.needsUpdate = true

						}

						// 窗玻璃更透，环境反射更强。
						if ( material.name === 'window glass' ) {

							material.color.set( '#2d424c' )
							material.transparent = true
							material.opacity = 0.3
							material.side = THREE.FrontSide
							material.metalness = 0
							material.roughness = 0
							material.transmission = 0
							material.specularIntensity = 1.5
							material.clearcoat = 3
							material.ior = 3
							material.envMapIntensity = 2
							material.needsUpdate = true

						}

					}

				} )
				site.add( gltf.scene )
				gltf.scene.updateMatrixWorld( true )
				gltf.scene.traverse( ( obj ) => {

					if ( ! obj.isMesh || ! obj.castShadow ) return
					if ( ! obj.geometry.boundingSphere ) obj.geometry.computeBoundingSphere()
					bound.copy( obj.geometry.boundingSphere ).applyMatrix4( obj.matrixWorld )
					// 离原点太远的网格关掉投射，阴影贴图留给近处。
					if ( bound.center.length() - bound.radius > SHADOW_CAST_RADIUS ) obj.castShadow = false

				} )
				markLoaded()

			}, undefined, () => {

				markLoaded()

			} )

		}

		/**
		 * 等 WebGPU 就绪后建后期链和循环。
		 * 模型加载和这一步并行：场地可以先往 scene 里加，通道在 init 之后才画。
		 */
		async function start() {

			if ( ! navigator.gpu ) {

				noteRef.current.textContent = '当前浏览器没有 WebGPU。'
				return

			}

			// 视图行进同时绑定 8 张存储纹理。设备默认上限是 4，这张显卡允许 8。
			renderer = new THREE.WebGPURenderer( {
				canvas,
				antialias: true,
				requiredLimits: { maxStorageTexturesPerShaderStage: 8 },
			} )
			renderer.library.addLight( SunLightNode, SunLight )
			renderer.shadowMap.enabled = true
			await renderer.init()
			if ( stopped ) {

				renderer.dispose()
				return

			}

			renderer.setPixelRatio( 1 )
			// 色调映射、sRGB、抖动在显示变换里做。渲染器保持线性，后期链最后一步再转显示。
			renderer.toneMapping = THREE.NoToneMapping
			const resize = () => {

				const w = canvas.clientWidth
				const h = Math.max( canvas.clientHeight, 1 )
				renderer.setSize( w, h, false )
				camera.aspect = w / h
				camera.updateProjectionMatrix()

			}
			onResize = resize
			resize()
			window.addEventListener( 'resize', onResize )

			// 只写深度，不跑光照，阴影就不会在这一遍里再画一次。法线由 GTAO 从深度图重建。
			// 预通道必须单采样：GTAO 的 textureGather 吃不了 MSAA 深度。场景通道继续跟渲染器走 4x MSAA。
			// 镂空用物体自己的 map.a 和 alphaTest；没有贴图时采样 1×1 白图，不会误裁。
			depthWhite = new THREE.DataTexture( new Uint8Array( [ 255, 255, 255, 255 ] ), 1, 1 )
			depthWhite.needsUpdate = true
			const depthMap = texture( depthWhite ).onObjectUpdate( ( { object }, self ) => {

				const material = object.material
				self.value = material && ! Array.isArray( material ) && material.map ? material.map : depthWhite

			} )
			const depthAlphaTest = uniform( 0 ).onObjectUpdate( ( { object }, self ) => {

				const material = object.material
				self.value = material && ! Array.isArray( material ) ? material.alphaTest || 0 : 0

			} )
			depthMaterial = new THREE.MeshBasicNodeMaterial()
			depthMaterial.colorWrite = false
			depthMaterial.toneMapped = false
			depthMaterial.colorNode = Fn( () => {

				depthMap.a.lessThan( depthAlphaTest ).discard()
				return vec4( 0, 0, 0, 1 )

			} )()
			prePass = pass( scene, camera, { samples: 1 } )
			prePass.overrideMaterial = depthMaterial
			aoPass = ao( prePass.getTextureNode( 'depth' ), null, camera )
			aoPass.resolutionScale = 0.5
			aoPass.useTemporalFiltering = false
			const aoAmount = uniform( 1 )

			// 后期链全是 TempNode，图里直接串起来。输出节点就是显示变换。
			// 深度预通道（单采样）→ 半分辨率 GTAO → 场景（MSAA，把 AO 混进内置环境光遮蔽）
			// → 泛光 → 雾（场景色 + 泛光）→ 炫光 → 自动曝光 → 调色 → 显示变换（ACES、sRGB、抖动）。
			// 雾色、密度、不透明度每帧按太阳写，构造时不必再给一遍。
			scenePass = pass( scene, camera )
			scenePass.contextNode = builtinAOContext( mix( float( 1 ), aoPass.getTextureNode().sample( screenUV ).r, aoAmount ) )
			const color = scenePass.getTextureNode( 'output' )
			const depth = scenePass.getTextureNode( 'depth' )
			bloom = new SceneBloom( color )
			fog = new SSF( {
				color: color.add( bloom ),
				depth,
				camera,
				fogHeight: 0.5,
				fogHeightFalloff: 0.05,
				fogStartDistance: 5,
				noiseStrength: 0.0015,
			} )
			glare = new Glare( {
				depth: scenePass.getTexture( 'depth' ),
				clouds,
				input: fog,
				camera,
			} )
			exposure = new AutoExposure( {
				color: bloom.meter,
				input: glare,
			} )
			pipeline = new THREE.RenderPipeline( renderer )
			// 关掉管线自带的输出变换，避免和 Display 里的 ACES、sRGB 叠两层。
			pipeline.outputColorTransform = false
			grade = new Grade( { input: exposure } )
			display = new Display( { input: grade } )
			pipeline.outputNode = display
			// 环境贴图只烤大气，不要视锥云和太阳盘。太阳方向转过约 0.004 弧度才重烤。
			environment = new SkyProbe( renderer, scene, sky )
			fps = createFps()

			/**
			 * 一帧顺序：
			 * 相机 / 演示时间轴 → 太阳月亮 → 阴影脏区 → 雾色密度 → 抄炫光和曝光的太阳 uniform
			 * → pipeline.render（各节点 updateBefore：泛光金字塔、雾绑相机、炫光写屏幕位置、调色和显示写帧序号）
			 * → compute 可见度和曝光适应（读的是这一帧刚画完的深度 / 计量图，下一帧才生效）。
			 */
			renderer.setAnimationLoop( ( time ) => {

				timer.update( time )
				const delta = timer.getDelta()
				if ( DEMO && contentReady && ! demoDone ) {

					demoT += delta
					runDemo( Math.min( demoT, DEMO_END ) )
					if ( demoT >= DEMO_END ) {

						demoDone = true
						camera.getWorldDirection( lookDir )
						orbit.target.copy( camera.position ).addScaledVector( lookDir, orbit.minDistance )
						orbit.update()
						orbit.enabled = true

					}

				} else if ( ! DEMO || demoDone ) {

					applyCam( settings.camera )
					if ( camMode === 'first' && look.isLocked ) {

						// WASD 走 xz，QE / 空格升降。y 只夹最低，不强制贴地，可以飞起来看。
						const speed = ( keys.has( 'ShiftLeft' ) || keys.has( 'ShiftRight' ) ? WALK_FAST : WALK ) * delta
						if ( keys.has( 'KeyW' ) ) look.moveForward( speed )
						if ( keys.has( 'KeyS' ) ) look.moveForward( - speed )
						if ( keys.has( 'KeyA' ) ) look.moveRight( - speed )
						if ( keys.has( 'KeyD' ) ) look.moveRight( speed )
						if ( keys.has( 'KeyE' ) || keys.has( 'Space' ) ) camera.position.y += speed
						if ( keys.has( 'KeyQ' ) ) camera.position.y -= speed
						camera.position.y = Math.max( EYE_MIN, camera.position.y )

					}

				}
				// 按相机刷新树林实例。没加载完时是 null。
				forest?.update( camera )
				const sun = settings.sun
				if ( sun.advance && ( ! DEMO || demoDone ) ) sun.setTime( ( sun.time + delta * sun.speed + 24 ) % 24 )
				applySun( sky, sun.time, sun.azimuth )
				sky.coverage = settings.coverage
				// EV 滑条写成显示变换的曝光：基准 0.55 乘 2^ev。调色还在线性 HDR 里，ACES 在 Display。
				display.exposure.value = 0.55 * 2 ** settings.ev
				aoAmount.value = settings.ao
				bloom.bloom.value = settings.bloom
				glare.strength.value = settings.glare
				grade.saturation.value = settings.saturation
				grade.contrast.value = settings.contrast
				grade.vignette.value = settings.vignette
				grade.grain.value = settings.grain
				placeMoon( sky )
				sky.update( renderer, delta, camera )
				environment.update( sky.sunDirection )
				// SunLight 没有 target，位置就是指向原点的方向。强度和颜色跟天空走。
				sunLight.position.copy( sky.sunDirection )
				sunLight.color.copy( sky.sunColor )
				sunLight.intensity = sky.sunIntensity
				moonLight.position.copy( sky.moonDirection )
				moonLight.color.copy( sky.moonColor )
				moonLight.intensity = sky.moonIntensity
				const sunLit = sky.sunIntensity >= sky.moonIntensity
				// 更亮的那盏出影。太阳月亮各记一份脏区，切回来不必立刻重画。
				sunLight.shadow.intensity = sunLit ? 1.0 : 0.0;
				moonLight.shadow.intensity = sunLit ? 0.0 : 1.0;
				const active = sunLit ? sunLight : moonLight
				const slot = sunLit ? sunShadow : moonShadow
				const dir = sunLit ? sky.sunDirection : sky.moonDirection
				const moved = camera.position.distanceToSquared( slot.pos ) > SHADOW_MOVE * SHADOW_MOVE
				const turned = camera.quaternion.angleTo( slot.quat ) > SHADOW_TURN
				if ( contentReady && ( ! slot.ready || dir.angleTo( slot.dir ) > SHADOW_DIR || moved || turned ) ) {

					active.shadow.needsUpdate = true
					slot.ready = true
					slot.dir.copy( dir )
					slot.pos.copy( camera.position )
					slot.quat.copy( camera.quaternion )

				}
				// 环境光颜色抄天空对地面的辐照（E/π），强度保持 0。
				ambient.color.copy( sky.skyIrradiance )
				const night = THREE.MathUtils.smoothstep( - sky.sunDirection.y, 0.02, 0.18 )
				fog.enabled = settings.fog
				fog.fogDensity = radiationFogDensity( sun.time )
				matchFogLuma( fogDay, sky.skyIrradiance )
				fogDay.lerp( FOG_REFERENCE, 0.55 )
				matchFogLuma( fogNight, sky.moonColor )
				fogNight.lerp( FOG_REFERENCE, 0.55 )
				fogTint.copy( fogDay ).lerp( fogNight, night )
				fogTint.multiplyScalar( THREE.MathUtils.lerp( 1, 0.22, night ) )
				fog.fogMaxOpacity = THREE.MathUtils.lerp( 0.35, 0.12, night )
				fog.fogColor = fogTint
				// 雾体颜色跟天空漫反射，亮度钉在参考雾色。逆光那一层用当前主光颜色。
				if ( sunLit ) fogInscatter.copy( sky.sunColor ).multiplyScalar( FOG_INSCATTER * ( sky.sunIntensity / SUN_ILLUMINANCE ) )
				else fogInscatter.copy( sky.moonColor ).multiplyScalar( FOG_INSCATTER * night )
				fog.directionalInscatteringColor = fogInscatter
				fog.inscatteringLightDirection = lightDirection.copy( sunLit ? sky.sunDirection : sky.moonDirection ).negate()
				// 炫光要太阳方向、颜色、亮度；自动曝光只要方向（夜里收倍数上限）。都不握天空节点，每帧抄。
				// 屏幕位置、圆盘半径、雾的相机矩阵在各自 updateBefore 里写，这里不必再调。
				glare.sunDirection.value.copy( sky.sunDirection )
				glare.sunColor.value.copy( sky.sunColor )
				glare.sunIntensity.value = sky.sunIntensity
				exposure.sunDirection.value.copy( sky.sunDirection )
				pipeline.render()
				// 可见度和曝光适应读的是这一帧刚画完的深度/计量图，下一帧才生效。
				renderer.compute( glare.visibility )
				renderer.compute( exposure.adapt )
				fps.frame( delta * 1000 )

			} )

		}

		start()

		return () => {

			// 异步加载回来先看 stopped，避免卸载后再往场景里塞东西。
			// 后期节点后建的先拆：显示 → 调色 → 曝光 → 炫光 → 雾 → 泛光，再拆通道和渲染器。
			stopped = true
			settings.dispose()
			canvas.removeEventListener( 'click', onCanvasClick )
			window.removeEventListener( 'keydown', onKeyDown )
			window.removeEventListener( 'keyup', onKeyUp )
			window.removeEventListener( 'blur', onBlur )
			orbit.dispose()
			look.dispose()
			timer.dispose()
			sky.dispose()
			clouds.dispose( renderer )
			disposeObject( site )
			forest?.dispose()
			display?.dispose()
			grade?.dispose()
			exposure?.dispose()
			glare?.dispose()
			fog?.dispose()
			environment?.dispose()
			fps?.dispose()
			bloom?.dispose()
			draco.dispose()
			if ( ! renderer ) return
			window.removeEventListener( 'resize', onResize )
			renderer.setAnimationLoop( null )
			pipeline?.dispose()
			aoPass?.dispose()
			prePass?.dispose()
			depthMaterial?.dispose()
			depthWhite?.dispose()
			scenePass?.dispose()
			renderer.dispose()

		}

	}, [] )

	return (
		<>
			<canvas ref={ canvasRef } />
			<p ref={ noteRef } />
		</>
	)

}
