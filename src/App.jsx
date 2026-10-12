import { useEffect, useRef } from 'react'
import * as THREE from 'three/webgpu'
import { cameraPosition, mix, normalize, pass, positionWorld, smoothstep, vec3 } from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { Sky } from './sky'
import { Glare } from './glare'
import { AutoExposure } from './exposure'

// 站在小岛边缘朝外看。
const STAND_Z = - 3200

function placeMoon( sky ) {

	const sun = sky.sunDirection

	sky.moonDirection.set( - sun.x, Math.abs( sun.y ) * 0.8 + 0.25, - sun.z ).normalize()

}

export default function App() {

	const canvasRef = useRef( null )
	const noteRef = useRef( null )

	useEffect( () => {

		const canvas = canvasRef.current
		const camera = new THREE.PerspectiveCamera( 50, 1, 0.1, 100 )
		camera.position.set( 0, 1.5, STAND_Z + 6 )
		const timer = new THREE.Timer()
		timer.connect( document )
		const scene = new THREE.Scene()
		const sky = new Sky( { sunDirection: new THREE.Vector3( 0.2, 0.45, - 1 ).normalize() } )
		sky.clouds = true
		sky.coverage = 0.49
		placeMoon( sky )
		scene.add( sky )

		const sunLight = new THREE.DirectionalLight( 0xffffff, 1 )
		const moonLight = new THREE.DirectionalLight( 0xffffff, 0 )
		const ambient = new THREE.AmbientLight( 0xffffff, 1 )
		scene.add( sunLight, sunLight.target, moonLight, moonLight.target, ambient )

		const sphereMaterial = new THREE.MeshLambertNodeMaterial()
		const sphere = new THREE.Mesh( new THREE.SphereGeometry( 0.7, 48, 32 ), sphereMaterial )
		sphere.position.set( 0, 0.7, STAND_Z )
		scene.add( sphere )

		const controls = new OrbitControls( camera, canvas )
		controls.target.set( 0, 3.2, STAND_Z )
		controls.enableDamping = true
		controls.minDistance = 2
		controls.maxDistance = 30
		controls.update()

		let renderer = null
		let pipeline = null
		let scenePass = null
		let stopped = false
		let onResize = null

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
			await renderer.init()
			if ( stopped ) {

				renderer.dispose()
				return

			}

			renderer.setPixelRatio( window.devicePixelRatio )
			// 自动曝光之后压高光。three 自带的 ACES Filmic，和 Tidewater 同一条曲线。
			renderer.toneMapping = THREE.ACESFilmicToneMapping
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
			} )
			const exposure = new AutoExposure( {
				color: scenePass.getTexture( 'output' ),
				sunDirection: sky.sunDirection,
			} )
			pipeline = new THREE.RenderPipeline( renderer )
			pipeline.outputNode = exposure.apply( glare.composite( scenePass ) )

			// 地平线以下换成海色，用来看平视时的曝光。天空本身在这块是黑的。
			const sea = vec3( 0.08, 0.28, 0.38 )
			let tinted = false

			renderer.setAnimationLoop( ( time ) => {

				timer.update( time )
				controls.update()
				placeMoon( sky )
				const delta = timer.getDelta()
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
				ambient.color.copy( sky.skyIrradiance )
				glare.update( renderer, camera, delta )
				exposure.update( delta )
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
			sphere.geometry.dispose()
			sphereMaterial.dispose()
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
