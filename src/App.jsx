import { useEffect, useRef } from 'react'
import * as THREE from 'three/webgpu'
import { cameraPosition, mix, normalize, pass, positionWorld, smoothstep, vec3 } from 'three/tsl'
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
import { grade, updateGrade } from './grade/grade'

// 开场机位，和页面里的轨道相机一致。
const HOME = { position: [ - 5, 1.2, 5 ], target: [ 0, 1, 0 ], fov: 45 }

// 整组模型先缩到 0.08，再挪到这个位置。道路在组内再抬高一点。
const SITE_FILES = [
	[ '/models/building-transformed.glb', 0 ],
	[ '/models/building_floor-transformed.glb', 0 ],
	[ '/models/building_trees-transformed.glb', 0 ],
	[ '/models/landscape-transformed.glb', 0 ],
	[ '/models/terrain.glb', 0 ],
	[ '/models/roads-transformed.glb', 0.019 ],
]

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
		const camera = new THREE.PerspectiveCamera( HOME.fov, 1, 0.1, 1000 )
		camera.position.set( ...HOME.position )
		const timer = new THREE.Timer()
		timer.connect( document )
		const scene = new THREE.Scene()
		const sky = new Sky( { sunDirection: new THREE.Vector3( 0.2, 0.45, - 1 ).normalize() } )
		const clouds = new SkyProField( sky.skyView, sky.transmittance )
		sky.clouds = clouds
		sky.coverage = 0.49
		placeMoon( sky )
		scene.add( sky )

		const sunLight = new SunLight( 0xffffff, 1 )
		const moonLight = new SunLight( 0xffffff, 0 )
		const ambient = new THREE.AmbientLight( 0xffffff, 1 )
		// 和 Planpoint 同一套：2048 贴图，近 30、远 80，bias -0.001，normalBias 0.05。
		// 同一时刻只让当前主光投射，避免太阳和月亮各渲一遍。
		sunLight.castShadow = true
		for ( const light of [ sunLight, moonLight ] ) {

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
		let scenePass = null
		let forest = null
		let bloom = null
		let stopped = false
		let onResize = null

		createForest().then( ( next ) => {

			if ( stopped ) {

				next.dispose()
				return

			}

			forest = next
			scene.add( forest.object )

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

					const material = obj.material
					if ( ! material || material.name !== 'TerrainCenterNew' ) return
					material.metalness = 0.3
					material.roughness = 0.95
					material.normalScale?.set( - 0.25, - 0.5 )

				} )
				site.add( gltf.scene )

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

			renderer.setPixelRatio( window.devicePixelRatio )
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

			// 炫光在场景画完之后叠加，不进天空颜色。
			scenePass = pass( scene, camera )
			const glare = new Glare( {
				depth: scenePass.getTexture( 'depth' ),
				sunDirection: sky.sunDirection,
				sunColor: sky.sunColorNode,
				sunIntensity: sky.sunIntensityNode,
				moonColor: sky.moonColorNode,
				moonIntensity: sky.moonIntensityNode,
				clouds,
			} )
			bloom = new SceneBloom( scenePass.getTextureNode( 'output' ) )
			const exposure = new AutoExposure( {
				color: bloom.meter,
				sunDirection: sky.sunDirection,
			} )
			pipeline = new THREE.RenderPipeline( renderer )
			pipeline.outputColorTransform = false
			pipeline.outputNode = grade( exposure.apply( glare.composite( scenePass.add( bloom ) ) ) )

			// 地平线以下换成海色，用来看平视时的曝光。天空本身在这块是黑的。
			const sea = vec3( 0.08, 0.28, 0.38 )
			let tinted = false

			renderer.setAnimationLoop( ( time ) => {

				timer.update( time )
				const delta = timer.getDelta()
				controls.update()
				forest?.update( camera )
				placeMoon( sky )
				sky.update( renderer, delta, camera )
				if ( ! tinted ) {

					const direction = normalize( positionWorld.sub( cameraPosition ) )
					const below = smoothstep( 0, - 0.02, direction.y )
					sky.material.colorNode = mix( sky.node, sea, below )
					tinted = true

				}
				sunLight.position.copy( sky.sunDirection )
				sunLight.color.copy( sky.sunColor )
				sunLight.intensity = sky.sunIntensity
				moonLight.position.copy( sky.moonDirection )
				moonLight.color.copy( sky.moonColor )
				moonLight.intensity = sky.moonIntensity
				const sunLit = sky.sunIntensity >= sky.moonIntensity
				sunLight.castShadow = sunLit
				moonLight.castShadow = ! sunLit
				ambient.color.copy( sky.skyIrradiance )
				glare.update( renderer, camera, delta )
				exposure.update( delta )
				updateGrade()
				pipeline.render()
				renderer.compute( glare.visibility )
				renderer.compute( exposure.adapt )

			} )

		}

		start()

		return () => {

			stopped = true
			controls.dispose()
			timer.dispose()
			sky.dispose()
			clouds.dispose( renderer )
			disposeObject( site )
			forest?.dispose()
			bloom?.dispose()
			draco.dispose()
			if ( ! renderer ) return
			window.removeEventListener( 'resize', onResize )
			renderer.setAnimationLoop( null )
			pipeline?.dispose()
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
