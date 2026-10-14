// 设置面板。太阳、大气和效果滑条接到场景。

const TOD = { w: 300, h: 148, cx: 150, cy: 100, rx: 118, up: 74, down: 32 }
const SKY_KEYS = [
	[ 0, '#040915', '#0b1730' ],
	[ 4.4, '#060d22', '#18244a' ],
	[ 5.3, '#18264f', '#b0616a' ],
	[ 6.1, '#35609e', '#f2a86e' ],
	[ 7.6, '#3d7cc2', '#a8d2ec' ],
	[ 12, '#2c74c6', '#c2e4f6' ],
	[ 16.3, '#3778be', '#b6daee' ],
	[ 17.4, '#3a5d9c', '#f1a25f' ],
	[ 18.1, '#262f5d', '#dc6a4e' ],
	[ 18.9, '#101a3c', '#473358' ],
	[ 19.8, '#050b1a', '#0f1b36' ],
	[ 24, '#040915', '#0b1730' ],
]
const PHASES = [ [ 4.8, '夜晚' ], [ 5.6, '黎明' ], [ 6.4, '日出' ], [ 7.2, '黄金时段' ], [ 10.5, '上午' ],
	[ 13.5, '正午' ], [ 16.8, '下午' ], [ 17.6, '黄金时段' ], [ 18.4, '日落' ], [ 19.3, '黄昏' ], [ 24, '夜晚' ] ]

const ICO = {
	sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
	cloud: '<path d="M7 18a4.5 4.5 0 1 1 .8-8.9A6 6 0 0 1 19 11a3.5 3.5 0 0 1-.5 7z"/>',
	sparkles: '<path d="M11 3.5 12.6 8.4 17.5 10l-4.9 1.6L11 16.5l-1.6-4.9L4.5 10l4.9-1.6z"/><path d="M18.5 14.5l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/>',
	camera: '<path d="M4 8h3l1.5-2.5h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
	chevron: '<path d="m6 9 6 6 6-6"/>',
	close: '<path d="M6 6l12 12M18 6 6 18"/>',
}

const clamp = ( v, a, b ) => Math.min( b, Math.max( a, v ) )
const hexToRgb = ( hex ) => {
	const n = parseInt( hex.slice( 1 ), 16 )
	return [ ( n >> 16 ) & 255, ( n >> 8 ) & 255, n & 255 ]
}
const mixHex = ( a, b, t ) => '#' + hexToRgb( a ).map( ( c, i ) => Math.round( c + ( hexToRgb( b )[ i ] - c ) * t ).toString( 16 ).padStart( 2, '0' ) ).join( '' )

function skyAt( hour ) {
	const hh = ( ( hour % 24 ) + 24 ) % 24
	for ( let i = 0; i < SKY_KEYS.length - 1; i ++ ) {
		const a = SKY_KEYS[ i ]
		const b = SKY_KEYS[ i + 1 ]
		if ( hh >= a[ 0 ] && hh <= b[ 0 ] ) {
			const t = ( hh - a[ 0 ] ) / ( ( b[ 0 ] - a[ 0 ] ) || 1 )
			const s = t * t * ( 3 - 2 * t )
			return [ mixHex( a[ 1 ], b[ 1 ], s ), mixHex( a[ 2 ], b[ 2 ], s ) ]
		}
	}
	return [ SKY_KEYS[ 0 ][ 1 ], SKY_KEYS[ 0 ][ 2 ] ]
}

function phaseAt( hh ) {
	for ( const [ end, name ] of PHASES ) if ( hh < end ) return name
	return '夜晚'
}

function todPoint( hours ) {
	const th = Math.PI - ( ( hours - 6 ) / 12 ) * Math.PI
	const s = Math.sin( th )
	return { x: TOD.cx + TOD.rx * Math.cos( th ), y: TOD.cy - ( s >= 0 ? TOD.up : TOD.down ) * s, elev: s }
}

function todHours( x, y ) {
	const nx = ( x - TOD.cx ) / TOD.rx
	const dy = TOD.cy - y
	const ny = dy >= 0 ? dy / TOD.up : dy / TOD.down
	return ( ( 6 + ( Math.PI - Math.atan2( ny, nx ) ) / Math.PI * 12 ) % 24 + 24 ) % 24
}

function ico( name ) {
	return `<svg class="tw-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ ICO[ name ] }</svg>`
}

function el( tag, cls, html ) {
	const node = document.createElement( tag )
	if ( cls ) node.className = cls
	if ( html ) node.innerHTML = html
	return node
}

function slider( label, value, min, max, step, format, log ) {
	const root = el( 'div', 'tw-ctl' )
	const shown = el( 'span', 'tw-value' )
	const fill = el( 'span', 'tw-slider-fill' )
	const thumb = el( 'span', 'tw-slider-thumb' )
	const rail = el( 'div', 'tw-slider', '<span class="tw-slider-rail"></span>' )
	rail.querySelector( '.tw-slider-rail' ).append( fill )
	rail.append( thumb )
	rail.tabIndex = 0
	root.append( el( 'div', 'tw-row', `<span class="tw-label"><span class="tw-label-text">${ label }</span></span>` ), rail )
	root.querySelector( '.tw-row' ).append( shown )
	const toT = ( v ) => log ? clamp( Math.log( v / min ) / Math.log( max / min ), 0, 1 ) : ( v - min ) / ( max - min || 1 )
	const fromT = ( t ) => log ? min * ( max / min ) ** t : min + t * ( max - min )
	let current = value
	const paint = ( next ) => {
		current = clamp( Math.round( next / step ) * step, min, max )
		const t = toT( current )
		fill.style.width = `${ t * 100 }%`
		thumb.style.left = `calc(7 * var(--tw-u) + (100% - 14 * var(--tw-u)) * ${ t })`
		shown.textContent = format( current )
		return current
	}
	paint( value )
	const at = ( event ) => {
		const rect = rail.getBoundingClientRect()
		current = paint( fromT( clamp( ( event.clientX - rect.left ) / rect.width, 0, 1 ) ) )
	}
	rail.addEventListener( 'pointerdown', ( event ) => {
		rail.setPointerCapture( event.pointerId )
		root.classList.add( 'is-dragging' )
		at( event )
	} )
	rail.addEventListener( 'pointermove', ( event ) => {
		if ( rail.hasPointerCapture( event.pointerId ) ) at( event )
	} )
	rail.addEventListener( 'pointerup', () => root.classList.remove( 'is-dragging' ) )
	return { root, paint, get value() { return current } }
}

function select( label, value, options ) {
	const root = el( 'div', 'tw-ctl tw-ctl-select is-segmented' )
	const head = el( 'div', 'tw-row' )
	head.append( el( 'span', 'tw-label', `<span class="tw-label-text">${ label }</span>` ) )
	const seg = el( 'div', 'tw-seg' )
	seg.setAttribute( 'role', 'radiogroup' )
	seg.setAttribute( 'aria-label', label )
	seg.style.setProperty( '--n', String( options.length ) )
	const ind = el( 'span', 'tw-seg-ind' )
	seg.append( ind )
	let current = value
	const btns = options.map( ( opt, i ) => {
		const button = el( 'button', 'tw-seg-btn' )
		button.type = 'button'
		button.setAttribute( 'role', 'radio' )
		button.textContent = opt.label
		button.addEventListener( 'click', () => paint( opt.value ) )
		seg.append( button )
		return button
	} )
	root.append( head, seg )
	const paint = ( next ) => {
		const i = options.findIndex( ( opt ) => opt.value === next )
		if ( i < 0 ) return current
		current = next
		btns.forEach( ( button, j ) => {
			button.setAttribute( 'aria-checked', String( j === i ) )
			button.tabIndex = j === i ? 0 : - 1
		} )
		ind.style.setProperty( '--i', String( i ) )
		return current
	}
	paint( value )
	return { root, paint, get value() { return current } }
}

function toggle( label, on, onChange ) {
	const root = el( 'button', 'tw-ctl tw-ctl-toggle' + ( on ? ' is-on' : '' ) )
	root.type = 'button'
	root.innerHTML = `<span class="tw-label"><span class="tw-label-text">${ label }</span></span><span class="tw-switch"><span class="tw-switch-knob"></span></span>`
	root.addEventListener( 'click', () => {
		on = ! on
		root.classList.toggle( 'is-on', on )
		onChange( on )
	} )
	return root
}

function folder( title, icon, open ) {
	const root = el( 'section', 'tw-folder' + ( open ? ' is-open' : '' ) )
	const head = el( 'button', 'tw-folder-head' )
	head.type = 'button'
	head.innerHTML = `${ ico( icon ) }<span class="tw-folder-title">${ title }</span>${ ico( 'chevron' ) }`
	head.querySelectorAll( '.tw-ico' )[ 0 ].classList.add( 'tw-folder-ico' )
	head.querySelectorAll( '.tw-ico' )[ 1 ].classList.add( 'tw-folder-chev' )
	const wrap = el( 'div', 'tw-folder-wrap' )
	const body = el( 'div', 'tw-folder-body' )
	wrap.append( body )
	head.addEventListener( 'click', () => root.classList.toggle( 'is-open' ) )
	root.append( head, wrap )
	return { root, body }
}

function timeDial( hour ) {
	const id = 'tw-tod-sky'
	const root = el( 'div', 'tw-ctl tw-ctl-time' )
	const clock = el( 'button', 'tw-value tw-clock' )
	clock.type = 'button'
	clock.tabIndex = - 1
	const row = el( 'div', 'tw-row', '<span class="tw-label"><span class="tw-label-text">时刻</span></span>' )
	row.append( clock )
	const dial = el( 'div', 'tw-tod' )
	let seed = 11
	const rnd = () => ( seed = ( seed * 16807 ) % 2147483647 ) / 2147483647
	let stars = ''
	for ( let i = 0; i < 26; i ++ ) {
		stars += `<circle cx="${ ( 6 + rnd() * ( TOD.w - 12 ) ).toFixed( 1 ) }" cy="${ ( 5 + rnd() * ( TOD.cy - 16 ) ).toFixed( 1 ) }" r="${ ( 0.45 + rnd() * 0.75 ).toFixed( 2 ) }" opacity="${ ( 0.35 + rnd() * 0.65 ).toFixed( 2 ) }"/>`
	}
	let ticks = ''
	for ( let hr = 0; hr < 24; hr += 3 ) {
		const p = todPoint( hr )
		ticks += `<circle cx="${ p.x.toFixed( 1 ) }" cy="${ p.y.toFixed( 1 ) }" r="${ hr % 6 ? 1.1 : 1.8 }"/>`
	}
	const labels = [ [ '06', TOD.cx - TOD.rx, TOD.cy + 13 ], [ '12', TOD.cx, TOD.cy - TOD.up - 7 ], [ '18', TOD.cx + TOD.rx, TOD.cy + 13 ], [ '00', TOD.cx, TOD.cy + TOD.down + 11 ] ]
		.map( ( [ t, x, y ] ) => `<text x="${ x }" y="${ y }">${ t }</text>` ).join( '' )
	dial.innerHTML = `<svg class="tw-tod-svg" viewBox="0 0 ${ TOD.w } ${ TOD.h }" role="slider" tabindex="0" aria-label="时刻">
		<defs>
			<linearGradient id="${ id }-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0"/><stop offset="1"/></linearGradient>
			<linearGradient id="${ id }-sea" x1="0" y1="0" x2="0" y2="1"><stop offset="0"/><stop offset="1" stop-color="#03101a"/></linearGradient>
			<radialGradient id="${ id }-glow"><stop offset="0" stop-color="#fff1d6" stop-opacity=".95"/><stop offset=".28" stop-color="#ffc98a" stop-opacity=".5"/><stop offset="1" stop-color="#ffb86b" stop-opacity="0"/></radialGradient>
			<radialGradient id="${ id }-haze"><stop offset="0" stop-color="#ffb070" stop-opacity=".8"/><stop offset="1" stop-color="#ff9a5a" stop-opacity="0"/></radialGradient>
			<linearGradient id="${ id }-glint" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffdcae" stop-opacity=".9"/><stop offset="1" stop-color="#ffdcae" stop-opacity="0"/></linearGradient>
			<mask id="${ id }-moon"><circle r="5.4" fill="#fff"/><circle cx="2.7" cy="-1.9" r="4.5" fill="#000"/></mask>
			<clipPath id="${ id }-clip"><rect width="${ TOD.w }" height="${ TOD.h }" rx="10"/></clipPath>
		</defs>
		<g clip-path="url(#${ id }-clip)">
			<rect width="${ TOD.w }" height="${ TOD.cy + 1 }" fill="url(#${ id }-sky)"/>
			<g class="tw-tod-stars" fill="#fff">${ stars }</g>
			<ellipse class="tw-tod-haze" cx="${ TOD.cx }" cy="${ TOD.cy }" rx="96" ry="30" fill="url(#${ id }-haze)"/>
			<rect y="${ TOD.cy }" width="${ TOD.w }" height="${ TOD.h - TOD.cy }" fill="url(#${ id }-sea)"/>
			<ellipse class="tw-tod-glint" cx="${ TOD.cx }" cy="${ TOD.cy + 8 }" rx="5" ry="10" fill="url(#${ id }-glint)"/>
			<line class="tw-tod-horizon" x1="0" y1="${ TOD.cy + 0.5 }" x2="${ TOD.w }" y2="${ TOD.cy + 0.5 }"/>
			<path class="tw-tod-path" d="M${ TOD.cx - TOD.rx } ${ TOD.cy } A${ TOD.rx } ${ TOD.up } 0 0 1 ${ TOD.cx + TOD.rx } ${ TOD.cy }"/>
			<path class="tw-tod-path is-down" d="M${ TOD.cx + TOD.rx } ${ TOD.cy } A${ TOD.rx } ${ TOD.down } 0 0 1 ${ TOD.cx - TOD.rx } ${ TOD.cy }"/>
			<g class="tw-tod-ticks">${ ticks }</g>
			<g class="tw-tod-labels">${ labels }</g>
			<g class="tw-tod-moon"><circle r="5.4" fill="#e3ecff" mask="url(#${ id }-moon)"/></g>
			<g class="tw-tod-sun"><circle r="24" fill="url(#${ id }-glow)"/><circle class="tw-tod-disc" r="7"/></g>
		</g>
		<text class="tw-tod-phase" x="14" y="23"></text>
	</svg>`
	const svg = dial.firstElementChild
	const q = ( s ) => svg.querySelector( s )
	const sun = q( '.tw-tod-sun' )
	const disc = q( '.tw-tod-disc' )
	const glow = sun.firstElementChild
	const starsEl = q( '.tw-tod-stars' )
	const moon = q( '.tw-tod-moon' )
	const haze = q( '.tw-tod-haze' )
	const glint = q( '.tw-tod-glint' )
	const phase = q( '.tw-tod-phase' )
	const stopZ = q( `#${ id }-sky stop:first-child` )
	const stopH = q( `#${ id }-sky stop:last-child` )
	const stopSea = q( `#${ id }-sea stop:first-child` )
	const track = el( 'div', 'tw-tod-track' )
	const bar = el( 'div', 'tw-tod-bar' )
	const stops = []
	for ( let hr = 0; hr <= 24; hr ++ ) {
		const [ zen, hor ] = skyAt( hr )
		stops.push( `${ mixHex( zen, hor, 0.55 ) } ${ ( hr / 24 * 100 ).toFixed( 2 ) }%` )
	}
	bar.style.background = `linear-gradient(90deg, ${ stops.join( ', ' ) })`
	const knob = el( 'div', 'tw-tod-knob' )
	const scale = el( 'div', 'tw-tod-scale' )
	scale.innerHTML = [ 0, 6, 12, 18, 24 ].map( ( hh ) => `<span style="left:${ hh / 24 * 100 }%">${ String( hh ).padStart( 2, '0' ) }</span>` ).join( '' )
	track.append( bar, knob, scale )
	root.append( row, dial, track )

	let current = hour
	const paint = ( next ) => {
		const hrs = current = ( ( next % 24 ) + 24 ) % 24
		const p = todPoint( hrs )
		const [ zen, hor ] = skyAt( hrs )
		const f = ( n ) => n.toFixed( 2 )
		stopZ.setAttribute( 'stop-color', zen )
		stopH.setAttribute( 'stop-color', hor )
		stopSea.setAttribute( 'stop-color', mixHex( hor, '#04121c', 0.72 ) )
		sun.setAttribute( 'transform', `translate(${ f( p.x ) } ${ f( p.y ) })` )
		disc.setAttribute( 'fill', p.elev < 0 ? '#ffd2a0' : mixHex( '#ffe2b4', '#fff8ec', clamp( p.elev * 2.2, 0, 1 ) ) )
		glow.setAttribute( 'opacity', clamp( ( p.elev + 0.12 ) / 0.25, 0.2, 1 ).toFixed( 3 ) )
		root.classList.toggle( 'is-night', p.elev < - 0.02 )
		const night = clamp( ( 0.08 - p.elev ) / 0.3, 0, 1 )
		starsEl.setAttribute( 'opacity', night.toFixed( 3 ) )
		const m = todPoint( hrs + 12 )
		moon.setAttribute( 'transform', `translate(${ f( m.x ) } ${ f( m.y ) })` )
		moon.setAttribute( 'opacity', ( night * 0.95 ).toFixed( 3 ) )
		const low = p.elev > - 0.25 ? ( 1 - Math.min( 1, Math.abs( p.elev ) / 0.6 ) ) ** 2 : 0
		haze.setAttribute( 'cx', f( p.x ) )
		haze.setAttribute( 'opacity', ( low * 0.85 ).toFixed( 3 ) )
		const gry = 6 + ( 1 - clamp( p.elev, 0, 1 ) ) * 16
		glint.setAttribute( 'cx', f( p.x ) )
		glint.setAttribute( 'cy', f( TOD.cy + gry * 0.95 ) )
		glint.setAttribute( 'rx', f( 2.5 + ( 1 - clamp( p.elev, 0, 1 ) ) * 3 ) )
		glint.setAttribute( 'ry', f( gry ) )
		glint.setAttribute( 'opacity', p.elev > 0 ? ( 0.85 * ( 1 - p.elev ) ** 1.5 ).toFixed( 3 ) : '0' )
		const hh = Math.floor( hrs )
		const mm = Math.round( ( hrs - hh ) * 60 ) % 60
		clock.textContent = `${ String( hh ).padStart( 2, '0' ) }:${ String( mm ).padStart( 2, '0' ) }`
		phase.textContent = phaseAt( hrs )
		knob.style.left = `${ ( hrs / 24 ) * 100 }%`
		return hrs
	}
	paint( hour )
	const drag = ( node, map ) => {
		node.addEventListener( 'pointerdown', ( event ) => {
			node.setPointerCapture( event.pointerId )
			root.classList.add( 'is-dragging' )
			current = paint( map( event ) )
		} )
		node.addEventListener( 'pointermove', ( event ) => {
			if ( node.hasPointerCapture( event.pointerId ) ) current = paint( map( event ) )
		} )
		node.addEventListener( 'pointerup', () => root.classList.remove( 'is-dragging' ) )
	}
	drag( svg, ( event ) => {
		const rect = svg.getBoundingClientRect()
		return todHours( ( event.clientX - rect.left ) / rect.width * TOD.w, ( event.clientY - rect.top ) / rect.height * TOD.h )
	} )
	drag( track, ( event ) => {
		const rect = bar.getBoundingClientRect()
		return clamp( ( event.clientX - rect.left ) / Math.max( 1, rect.width ), 0, 1 ) * 24
	} )
	return { root, paint, get value() { return current } }
}

export function createSettings() {
	const root = el( 'div', 'tw-ui' )
	root.dataset.panel = 'closed'
	const rail = el( 'div', 'tw-rail tw-glass tw-interactive' )
	const tabs = el( 'div', 'tw-rail-tabs' )
	const panel = el( 'div', 'tw-panel tw-glass tw-interactive' )
	const head = el( 'div', 'tw-panel-head' )
	const title = el( 'div', 'tw-panel-title' )
	const close = el( 'button', 'tw-icon-btn', ico( 'close' ) )
	close.type = 'button'
	close.setAttribute( 'aria-label', '关闭' )
	head.append( title, close )
	const strip = el( 'div', 'tw-tabs' )
	const pages = el( 'div', 'tw-pages' )
	panel.append( head, strip, pages )

	const pagesByName = {}
	let current = 'sky'

	function page( name, label, icon ) {
		const button = el( 'button', 'tw-rail-btn', ico( icon ) )
		button.type = 'button'
		button.setAttribute( 'aria-label', label )
		const tab = el( 'button', 'tw-tab', ico( icon ) + `<span class="tw-tab-label">${ label }</span>` )
		tab.type = 'button'
		const body = el( 'div', 'tw-page' )
		tabs.append( button )
		strip.append( tab )
		pages.append( body )
		pagesByName[ name ] = { button, tab, body, label }
		button.addEventListener( 'click', () => open( name ) )
		tab.addEventListener( 'click', () => open( name ) )
		return body
	}

	function open( name ) {
		current = name
		root.dataset.panel = 'open'
		title.textContent = pagesByName[ name ].label
		for ( const key in pagesByName ) {
			const item = pagesByName[ key ]
			const on = key === name
			item.button.classList.toggle( 'is-active', on )
			item.tab.classList.toggle( 'is-active', on )
			item.body.hidden = ! on
		}
	}

	function shut() {
		root.dataset.panel = 'closed'
	}

	close.addEventListener( 'click', shut )

	const sky = page( 'sky', '天空', 'sun' )
	const sunFolder = folder( '太阳', 'sun', true )
	const dial = timeDial( 10.0 )
	const azimuth = slider( '太阳方位', 0, - 180, 180, 1, ( v ) => `${ Math.round( v ) }°` )
	const speed = slider( '时间速度', 0.02, 0.002, 1, 0.001, ( v ) => `${ v.toFixed( 3 ) } 时/秒`, true )
	const sun = {
		advance: false,
		get time() { return dial.value },
		get azimuth() { return azimuth.value },
		get speed() { return speed.value },
		setTime( hours ) { dial.paint( hours ) },
	}
	speed.root.hidden = true
	sunFolder.body.append( dial.root, azimuth.root )
	sunFolder.body.append( toggle( '推进时间', false, ( on ) => {
		sun.advance = on
		speed.root.hidden = ! on
	} ) )
	sunFolder.body.append( speed.root )
	const air = folder( '大气', 'cloud', true )
	const coverage = slider( '云量', 0.49, 0, 1, 0.01, ( v ) => `${ Math.round( v * 100 ) }%` )
	const ev = slider( '曝光', 0, - 3, 3, 0.1, ( v ) => `${ v > 0 ? '+' : '' }${ v.toFixed( 1 ) } EV` )
	const fog = { on: true }
	air.body.append( coverage.root )
	air.body.append( toggle( '雾', true, ( on ) => { fog.on = on } ) )
	air.body.append( ev.root )
	sky.append( sunFolder.root, air.root )

	const cam = page( 'camera', '相机', 'camera' )
	const view = folder( '视图', 'camera', true )
	const camera = select( '相机', 'orbit', [
		{ label: '第一人称', value: 'first' },
		{ label: '轨道', value: 'orbit' },
	] )
	view.body.append( camera.root )
	cam.append( view.root )

	const post = page( 'post', '效果', 'sparkles' )
	const fx = folder( '效果', 'sparkles', true )
	const ao = slider( '环境光遮蔽', 1, 0, 1.5, 0.01, ( v ) => v.toFixed( 2 ) )
	const bloom = slider( '泛光', 0.05, 0, 0.3, 0.005, ( v ) => v.toFixed( 3 ) )
	fx.body.append( ao.root )
	const glare = slider( '眩光', 1, 0, 2, 0.05, ( v ) => v.toFixed( 2 ) )
	fx.body.append( bloom.root )
	const saturation = slider( '饱和度', 1.06, 0.5, 1.5, 0.01, ( v ) => v.toFixed( 2 ) )
	const contrast = slider( '对比度', 1.04, 0.8, 1.3, 0.01, ( v ) => v.toFixed( 2 ) )
	const vignette = slider( '暗角', 0.28, 0, 1, 0.01, ( v ) => v.toFixed( 2 ) )
	const grain = slider( '颗粒', 0.012, 0, 0.06, 0.001, ( v ) => v.toFixed( 3 ) )
	fx.body.append( glare.root )
	fx.body.append( saturation.root )
	fx.body.append( contrast.root )
	fx.body.append( vignette.root )
	fx.body.append( grain.root )
	post.append( fx.root )

	const sep = el( 'div', 'tw-rail-sep' )
	rail.append( tabs, sep )
	root.append( rail, panel )
	document.body.append( root )
	open( 'sky' )
	shut()
	void current

	return {
		sun,
		get coverage() { return coverage.value },
		setCoverage( value ) { coverage.paint( value ) },
		get fog() { return fog.on },
		get ev() { return ev.value },
		get ao() { return ao.value },
		get bloom() { return bloom.value },
		get glare() { return glare.value },
		get saturation() { return saturation.value },
		get contrast() { return contrast.value },
		get vignette() { return vignette.value },
		get grain() { return grain.value },
		get camera() { return camera.value },
		setCamera( mode ) { camera.paint( mode ) },
		open,
		dispose() { root.remove() },
	}
}
