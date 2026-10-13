// 左上角帧率：250ms 刷新数字，火花线记每 100ms 里最慢的一帧。

const RGB = { good: '95, 227, 212', warn: '255, 184, 107', bad: '255, 122, 133' }

export function createFps() {

	const root = document.createElement( 'div' )
	root.className = 'tw-stats tw-glass'
	root.dataset.level = 'good'
	root.setAttribute( 'aria-hidden', 'true' )
	root.innerHTML = `
		<div class="tw-stats-main"><span class="tw-fps">--</span><span class="tw-fps-unit">fps</span></div>
		<canvas class="tw-spark"></canvas>
		<div class="tw-stats-sub"><span class="tw-kv"><span class="tw-k">frame</span><span class="tw-v">--</span></span></div>`
	document.body.append( root )

	const fpsEl = root.querySelector( '.tw-fps' )
	const msEl = root.querySelector( '.tw-v' )
	const spark = root.querySelector( '.tw-spark' )
	const samples = []
	let worst = 0
	let bucket = performance.now()
	let shown = bucket
	let sum = 0
	let count = 0
	let level = 'good'
	let text = ''
	let skip = true

	function frame( ms ) {

		if ( ! ( ms > 0 ) ) return
		if ( skip ) {

			skip = false
			return

		}

		const now = performance.now()
		worst = Math.max( worst, ms )
		sum += ms
		count ++
		if ( now - bucket >= 100 && worst > 0 ) {

			bucket = now
			samples.push( worst )
			if ( samples.length > 60 ) samples.shift()
			worst = 0
			draw()

		}

		if ( now - shown < 250 || count === 0 ) return
		shown = now
		const fps = 1000 / ( sum / count )
		sum = 0
		count = 0
		const nextText = String( Math.round( fps ) )
		if ( nextText !== text ) {

			text = nextText
			fpsEl.textContent = nextText

		}

		const nextLevel = fps >= 50 ? 'good' : fps >= 30 ? 'warn' : 'bad'
		if ( nextLevel !== level ) {

			level = nextLevel
			root.dataset.level = nextLevel
			draw()

		}

		msEl.textContent = `${ ( 1000 / fps ).toFixed( 1 ) } ms`

	}

	function draw() {

		const w = spark.clientWidth
		const h = spark.clientHeight
		if ( ! w || ! h || samples.length < 2 ) return
		const dpr = Math.min( 2, window.devicePixelRatio || 1 )
		const W = Math.round( w * dpr )
		const H = Math.round( h * dpr )
		if ( spark.width !== W || spark.height !== H ) {

			spark.width = W
			spark.height = H

		}

		const g = spark.getContext( '2d' )
		g.setTransform( dpr, 0, 0, dpr, 0, 0 )
		g.clearRect( 0, 0, w, h )
		const top = Math.max( 34, ...samples ) * 1.1
		const Y = ( v ) => h - 1.5 - ( Math.min( v, top ) / top ) * ( h - 4 )
		const X = ( i ) => w - 2 - ( samples.length - 1 - i ) * ( ( w - 4 ) / 59 )
		g.strokeStyle = 'rgba(200, 230, 240, 0.18)'
		g.lineWidth = 1
		g.setLineDash( [ 2, 3 ] )
		g.beginPath()
		const yr = Math.round( Y( 1000 / 60 ) ) + 0.5
		g.moveTo( 0, yr )
		g.lineTo( w, yr )
		g.stroke()
		g.setLineDash( [] )

		const rgb = RGB[ level ]
		const trace = () => {

			g.beginPath()
			samples.forEach( ( v, i ) => ( i ? g.lineTo( X( i ), Y( v ) ) : g.moveTo( X( i ), Y( v ) ) ) )

		}

		const grad = g.createLinearGradient( 0, 0, 0, h )
		grad.addColorStop( 0, `rgba(${ rgb }, 0.32)` )
		grad.addColorStop( 1, `rgba(${ rgb }, 0)` )
		trace()
		g.lineTo( X( samples.length - 1 ), h )
		g.lineTo( X( 0 ), h )
		g.closePath()
		g.fillStyle = grad
		g.fill()
		trace()
		g.strokeStyle = `rgba(${ rgb }, 0.95)`
		g.lineWidth = 1.25
		g.lineJoin = 'round'
		g.stroke()
		const last = samples.length - 1
		g.fillStyle = `rgb(${ rgb })`
		g.beginPath()
		g.arc( X( last ), Y( samples[ last ] ), 1.8, 0, Math.PI * 2 )
		g.fill()

	}

	return {
		frame,
		dispose() {

			root.remove()

		},
	}

}
