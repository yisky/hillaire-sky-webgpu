import { useEffect, useRef } from 'react'
import * as THREE from 'three/webgpu'
import { builtinAOContext, float, Fn, mix, pass, screenUV, texture, uniform, vec4 } from 'three/tsl'
import { ao } from 'three/addons/tsl/display/GTAONode.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js'
import { SunLight } from 'three/addons/lights/SunLight.js'
import { SunLightNode } from 'three/addons/lights/SunLightNode.js'
import { Sky } from './sky'
import { SkyProField } from './sky/clouds/skypro'
import { Glare } from './glare'
import { AutoExposure } from './exposure'
import { createForest } from './trees'
import { SceneBloom } from './bloom/Bloom'
import { exposure as gradeExposure, grade, gradeParams, updateGrade } from './grade/grade'
import { SSF } from './ssf/SSF'
import { SkyEnvironment } from './env/Environment'
import { createFps } from './ui/fps'
import { createSettings } from './ui/settings'

// 开场机位，和页面里的轨道相机一致。
const HOME = { position: [ - 5, 1.2, 5 ], target: [ 0, 1, 0 ], fov: 45 }

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

// Tidewater 的太阳位置：北纬 24°、赤纬 6°。返回指向太阳的方向，+x 东、−z 北、+y 上。
function sunDirectionFromTime( hours, out ) {

	const phi = THREE.MathUtils.degToRad( 24 )
	const dec = THREE.MathUtils.degToRad( 6 )
	const hourAngle = THREE.MathUtils.degToRad( ( hours - 12 ) * 15 )
	const east = - Math.cos( dec ) * Math.sin( hourAngle )
	const north = Math.cos( phi ) * Math.sin( dec ) - Math.sin( phi ) * Math.cos( dec ) * Math.cos( hourAngle )
	const up = Math.sin( phi ) * Math.sin( dec ) + Math.cos( phi ) * Math.cos( dec ) * Math.cos( hourAngle )
	return out.set( east, up, - north ).normalize()

}

function applySun( sky, hours, azimuth ) {

	sunDirectionFromTime( hours, sky.sunDirection ).applyAxisAngle( SUN_UP, THREE.MathUtils.degToRad( azimuth ) )

}

const FOG_DENSITY_NOON = 0.005
const FOG_DENSITY_HORIZON = 0.012
const FOG_DAWN = 5.8
const FOG_CLEAR = 10
const FOG_DUSK = 18.2
const FOG_GATHER = 3.5
const FOG_REFERENCE = new THREE.Color( '#d0dce8' )
const FOG_LUMA = 0.2126 * FOG_REFERENCE.r + 0.7152 * FOG_REFERENCE.g + 0.0722 * FOG_REFERENCE.b

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

function placeMoon( sky ) {

	const sun = sky.sunDirection

	sky.moonDirection.set( - sun.x, Math.abs( sun.y ) * 0.8 + 0.25, - sun.z ).normalize()

}

function disposeObject( root ) {

	root.traverse( ( obj ) => {

		obj.geometry?.dispose()
		const materials = obj.material
		if ( ! materials ) return
		for ( const material of Array.isArray( materials ) ? materials : [ materials ] ) material.dispose()

	} )

}

export default function App() {

	const canvasRef = useRef( null )
	const noteRef = useRef( null )

	useEffect( () => {

		const canvas = canvasRef.current
		const settings = createSettings()
		const camera = new THREE.PerspectiveCamera( HOME.fov, 1, 0.1, 1000 )
		camera.position.set( ...HOME.position )
		const timer = new THREE.Timer()
		timer.connect( document )
		const scene = new THREE.Scene()
		const sunDir = new THREE.Vector3()
		sunDirectionFromTime( settings.sun.time, sunDir ).applyAxisAngle( SUN_UP, THREE.MathUtils.degToRad( settings.sun.azimuth ) )
		const sky = new Sky( { sunDirection: sunDir } )
		const clouds = new SkyProField( sky.skyView, sky.transmittance )
		sky.clouds = clouds
		sky.coverage = settings.coverage
		placeMoon( sky )
		scene.add( sky )

		const sunLight = new SunLight( 0xffffff, 1 )
		const moonLight = new SunLight( 0xffffff, 0 )
		// 环境贴图已经提供天空的漫反射。这里再给满，楼会亮两遍。
		const ambient = new THREE.AmbientLight( 0xffffff, 0 )
		// 和 Planpoint 同一套：2048 贴图，近 30、远 80，bias -0.001，normalBias 0.05。
		// 同一时刻只让当前主光投射，避免太阳和月亮各渲一遍。阴影不自动更新。
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
			//light.shadow.intensity = 0.95

		}
		scene.add( sunLight, moonLight, ambient )

		const site = new THREE.Group()
		site.scale.setScalar( 0.08 )
		site.position.set( 5.8, 0, 1 )
		scene.add( site )

		const draco = new DRACOLoader()
		draco.setDecoderPath( '/draco/' )
		const loader = new GLTFLoader()
		loader.setDRACOLoader( draco )

		const controls = new OrbitControls( camera, canvas )
		controls.target.set( ...HOME.target )
		// controls.enablePan = false
		// controls.screenSpacePanning = false
		// controls.minPolarAngle = 1.1
		// controls.maxPolarAngle = Math.PI / 2 - 0.05
		controls.enableDamping = false
		controls.zoomSpeed = 0.5
		controls.minDistance = 4
		controls.maxDistance = 25
		controls.update()

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
		let environment = null
		let fps = null
		const lightDirection = new THREE.Vector3()
		const fogDay = new THREE.Color()
		const fogNight = new THREE.Color()
		const fogTint = new THREE.Color()
		let stopped = false
		let onResize = null
		let loadsLeft = SITE_FILES.length + 1
		let contentReady = false
		const markLoaded = () => {

			loadsLeft --
			if ( loadsLeft === 0 ) contentReady = true

		}
		const shadowSlot = () => ( {
			dir: new THREE.Vector3(),
			pos: new THREE.Vector3(),
			quat: new THREE.Quaternion(),
			ready: false,
		} )
		const sunShadow = shadowSlot()
		const moonShadow = shadowSlot()
		const bound = new THREE.Sphere()

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
				gltf.scene.traverse( ( obj ) => {

					if ( obj.isMesh ) {

						obj.castShadow = true
						obj.receiveShadow = true

					}

					const materials = obj.material
					if ( ! materials ) return
					for ( const material of Array.isArray( materials ) ? materials : [ materials ] ) {

						if ( material.name === 'TerrainCenterNew' ) {

							material.metalness = 0.3
							material.roughness = 0.95
							material.normalScale?.set( - 0.25, - 0.5 )

						}

						// Planpoint 把模型里被遮罩丢掉的阳台玻璃改成半透明反射。
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
					if ( bound.center.length() - bound.radius > SHADOW_CAST_RADIUS ) obj.castShadow = false

				} )
				markLoaded()

			}, undefined, () => {

				markLoaded()

			} )

		}

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
			// ACES、固定曝光 0.55、sRGB 和抖动都在 grade 里做，这里不再套一层。
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
			depthWhite = new THREE.DataTexture( new Uint8Array( [ 255, 255, 255, 255 ] ), 1, 1 )
			depthWhite.needsUpdate = true
			const depthMap = uniform( depthWhite ).onObjectUpdate( ( { object }, self ) => {

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

				texture( depthMap ).a.lessThan( depthAlphaTest ).discard()
				return vec4( 0, 0, 0, 1 )

			} )()
			prePass = pass( scene, camera )
			prePass.overrideMaterial = depthMaterial
			aoPass = ao( prePass.getTextureNode( 'depth' ), null, camera )
			aoPass.resolutionScale = 0.5
			aoPass.useTemporalFiltering = false
			const aoAmount = uniform( 1 )

			scenePass = pass( scene, camera )
			scenePass.contextNode = builtinAOContext( mix( float( 1 ), aoPass.getTextureNode().sample( screenUV ).r, aoAmount ) )
			const color = scenePass.getTextureNode( 'output' )
			const depth = scenePass.getTextureNode( 'depth' )
			const glare = new Glare( {
				depth: scenePass.getTexture( 'depth' ),
				sunDirection: sky.sunDirection,
				sunColor: sky.sunColorNode,
				sunIntensity: sky.sunIntensityNode,
				moonColor: sky.moonColorNode,
				moonIntensity: sky.moonIntensityNode,
				clouds,
			} )
			bloom = new SceneBloom( color )
			fog = new SSF( color.add( bloom ), depth, {
				fogColor: '#d0dce8',
				fogMaxOpacity: 0.35,
				fogDensity: 0.005,
				fogHeight: 0.5,
				fogHeightFalloff: 0.05,
				fogStartDistance: 5,
				noiseStrength: 0.0015,
				directionalInscatteringColor: '#d0dce8',
			} )
			fog.setCamera( camera )
			const exposure = new AutoExposure( {
				color: bloom.meter,
				sunDirection: sky.sunDirection,
			} )
			pipeline = new THREE.RenderPipeline( renderer )
			pipeline.outputColorTransform = false
			pipeline.outputNode = grade( exposure.apply( glare.composite( fog.outputNode ) ) )
			environment = new SkyEnvironment( renderer, scene, sky )
			fps = createFps()

			renderer.setAnimationLoop( ( time ) => {

				timer.update( time )
				const delta = timer.getDelta()
				controls.update()
				forest?.update( camera )
				const sun = settings.sun
				if ( sun.advance ) sun.setTime( ( sun.time + delta * sun.speed + 24 ) % 24 )
				applySun( sky, sun.time, sun.azimuth )
				sky.coverage = settings.coverage
				gradeExposure.value = 0.55 * 2 ** settings.ev
				aoAmount.value = settings.ao
				bloom.bloom.value = settings.bloom
				glare.strength.value = settings.glare
				gradeParams.saturation.value = settings.saturation
				gradeParams.contrast.value = settings.contrast
				gradeParams.vignette.value = settings.vignette
				gradeParams.grain.value = settings.grain
				placeMoon( sky )
				sky.update( renderer, delta, camera )
				environment.update( sky.sunDirection )
				sunLight.position.copy( sky.sunDirection )
				sunLight.color.copy( sky.sunColor )
				sunLight.intensity = sky.sunIntensity
				moonLight.position.copy( sky.moonDirection )
				moonLight.color.copy( sky.moonColor )
				moonLight.intensity = sky.moonIntensity
				const sunLit = sky.sunIntensity >= sky.moonIntensity
				// sunLight.castShadow = sunLit
				// moonLight.castShadow = ! sunLit
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
				fog.directionalInscatteringColor = fogTint
				fog.inscatteringLightDirection = lightDirection.copy( sunLit ? sky.sunDirection : sky.moonDirection ).negate()
				glare.update( renderer, camera, delta )
				exposure.update( delta )
				updateGrade()
				pipeline.render()
				renderer.compute( glare.visibility )
				renderer.compute( exposure.adapt )
				fps.frame( delta * 1000 )

			} )

		}

		start()

		return () => {

			stopped = true
			settings.dispose()
			controls.dispose()
			timer.dispose()
			sky.dispose()
			clouds.dispose( renderer )
			disposeObject( site )
			forest?.dispose()
			fog?.material.dispose()
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
