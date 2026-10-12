/**
 * SkyPro “Partly cloudy” 的手写 WGSL。
 *
 * 密度、行进和天边消融照 Tidewater `src/sky/SkyProClouds.js`。
 * 云底 4000 米，厚 5200 米。25 公里到 45 公里的云混进天空色。
 * 形状噪声铺成 512² 的二维图，用 `textureLoad` 做三线性插值。
 * 查找表是存储纹理。
 */

import { wgslFn } from 'three/tsl';

const skyproLib = wgslFn( /* wgsl */`
fn scSat(x: f32) -> f32 {
	return clamp(x, 0.0, 1.0);
}

fn scImod(i: i32, n: i32) -> i32 {
	let m = i % n;
	return select(m + n, m, m >= 0);
}

fn scLoad2f(tex: texture_storage_2d<rgba16float, read>, uv: vec2f) -> vec4f {
	let size = vec2i(textureDimensions(tex));
	let p = clamp(uv, vec2f(0.0), vec2f(1.0)) * vec2f(size) - vec2f(0.5);
	let i = vec2i(floor(p));
	let f = fract(p);
	var acc = vec4f(0.0);
	for (var y = 0; y <= 1; y++) {
		for (var x = 0; x <= 1; x++) {
			let c = clamp(i + vec2i(x, y), vec2i(0), size - vec2i(1));
			let w = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
			acc += textureLoad(tex, vec2u(c)) * w;
		}
	}
	return acc;
}

fn scTransmittance(tex: texture_storage_2d<rgba16float, read>, radius: f32, mu: f32) -> vec3f {
	let Rg = 6360.0;
	let Rt = 6460.0;
	let H = sqrt(Rt * Rt - Rg * Rg);
	let rho = sqrt(max(radius * radius - Rg * Rg, 0.0));
	let disc = radius * radius * (mu * mu - 1.0) + Rt * Rt;
	let d = max(-radius * mu + sqrt(max(disc, 0.0)), 0.0);
	let dMin = Rt - radius;
	let dMax = rho + H;
	let xMu = (d - dMin) / (dMax - dMin);
	let xR = rho / H;
	let uv = vec2f((xMu + 0.5 / 256.0) * (256.0 / 257.0), (xR + 0.5 / 64.0) * (64.0 / 65.0));
	return scLoad2f(tex, uv).rgb;
}

fn scSkyLut(tex: texture_storage_2d<rgba16float, read>, viewHeight: f32, sunDir: vec3f, rd: vec3f) -> vec3f {
	let Rg = 6360.0;
	let horizon = sqrt(max(viewHeight * viewHeight - Rg * Rg, 0.0));
	let beta = acos(clamp(horizon / max(viewHeight, 1e-3), -1.0, 1.0));
	let zenithToHorizon = 3.14159265 - beta;
	let viewZenith = acos(clamp(rd.y, -1.0, 1.0));
	let above = (1.0 - sqrt(max(1.0 - viewZenith / max(zenithToHorizon, 1e-3), 0.0))) * 0.5;
	let below = sqrt(max((viewZenith - zenithToHorizon) / max(beta, 1e-3), 0.0)) * 0.5 + 0.5;
	let v = select(below, above, viewZenith < zenithToHorizon);
	let sunH = normalize(sunDir.xz + vec2f(1e-5, 0.0));
	let viewH = normalize(rd.xz + vec2f(1e-5, 0.0));
	let u = sqrt(clamp(dot(sunH, viewH) * -0.5 + 0.5, 0.0, 1.0));
	let uv = vec2f(u * (191.0 / 192.0) + 0.5 / 192.0, v * (107.0 / 108.0) + 0.5 / 108.0);
	return scLoad2f(tex, uv).rgb;
}

fn scSky(tex: texture_storage_2d<rgba16float, read>, viewHeight: f32, sunDir: vec3f, rd: vec3f, skyIrr: vec3f, night: f32) -> vec3f {
	return scSkyLut(tex, viewHeight, sunDir, rd) + skyIrr * night;
}

fn scDirect(trans: texture_storage_2d<rgba16float, read>, sunDir: vec3f) -> vec3f {
	return scTransmittance(trans, 6364.0, sunDir.y) * 11.0;
}

fn scHG(mu: f32, g: f32) -> f32 {
	let d = max(1.0 + g * g - 2.0 * g * mu, 0.001);
	return (1.0 - g * g) / (4.0 * 3.14159265 * d * sqrt(d));
}

fn scPhases(mu: f32) -> vec3f {
	return vec3f(scHG(mu, 0.8), scHG(mu, 0.4), scHG(mu, 0.2)) * 0.8
		+ vec3f(scHG(mu, -0.2), scHG(mu, -0.1), scHG(mu, -0.05)) * 0.2;
}

fn scHash33(cell: vec3u) -> vec3u {
	var p = cell * 1664525u + 1013904223u;
	p.x += p.y * p.z; p.y += p.z * p.x; p.z += p.x * p.y;
	p = p ^ (p >> vec3u(16u));
	p.x += p.y * p.z; p.y += p.z * p.x; p.z += p.x * p.y;
	return p;
}

fn scGradient(cell: vec3i, period: i32, offset: vec3f) -> f32 {
	let wrapped = vec3u(((cell % vec3i(period)) + vec3i(period)) % vec3i(period));
	let h = (scHash33(wrapped).x >> 24u) & 15u;
	let u = select(offset.y, offset.x, h < 8u);
	let v = select(select(offset.z, offset.x, h == 12u || h == 14u), offset.y, h < 4u);
	return select(-u, u, (h & 1u) == 0u) + select(-v, v, (h & 2u) == 0u);
}

fn scPerlin(p: vec3f, period: i32) -> f32 {
	let cell = vec3i(floor(p));
	let fr = fract(p);
	let w = fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0);
	return mix(mix(mix(scGradient(cell, period, fr), scGradient(cell + vec3i(1, 0, 0), period, fr - vec3f(1.0, 0.0, 0.0)), w.x),
		mix(scGradient(cell + vec3i(0, 1, 0), period, fr - vec3f(0.0, 1.0, 0.0)), scGradient(cell + vec3i(1, 1, 0), period, fr - vec3f(1.0, 1.0, 0.0)), w.x), w.y),
		mix(mix(scGradient(cell + vec3i(0, 0, 1), period, fr - vec3f(0.0, 0.0, 1.0)), scGradient(cell + vec3i(1, 0, 1), period, fr - vec3f(1.0, 0.0, 1.0)), w.x),
			mix(scGradient(cell + vec3i(0, 1, 1), period, fr - vec3f(0.0, 1.0, 1.0)), scGradient(cell + vec3i(1, 1, 1), period, fr - vec3f(1.0, 1.0, 1.0)), w.x), w.y), w.z);
}

fn scFbm(uv: vec2f, octaves: f32, slice: f32) -> f32 {
	var frequency = max(1.0, round(octaves));
	var weight = 0.5;
	var sum = 0.0;
	var weights = 0.0;
	let count = u32(select(5.0, 6.0, slice > 0.5));
	for (var i = 0u; i < count; i++) {
		sum += scPerlin(vec3f(uv, slice * 13.37 + 0.5) * frequency, i32(frequency)) * weight;
		weights += weight;
		weight *= 0.5;
		frequency *= 2.0;
	}
	return sum / max(weights, 0.001) * 0.5 + 0.5;
}

fn scWeather(uv: vec2f) -> vec4f {
	let mass = clamp((scFbm(uv, 4.0, 0.0) - 0.5) * 1.32 + 0.5, 0.0, 1.0);
	let detail = (scFbm(uv, 6.0, 1.0) * 2.0 - 1.0) * 0.13;
	return vec4f(clamp(mass + detail + 0.26 - 0.5, 0.0, 1.0), 0.0, 0.0, 1.0);
}

fn scBoundsAt(id: vec2u, weather: texture_storage_2d<rgba8unorm, read>) -> f32 {
	let sourceSize = vec2i(textureDimensions(weather));
	let scale = vec2f(sourceSize) / vec2f(64.0);
	let lo = vec2i(floor(vec2f(id) * scale - 0.5));
	let hi = vec2i(ceil(vec2f(id + vec2u(1u)) * scale - 0.5));
	var maximum = 0.0;
	for (var y = lo.y; y <= hi.y; y++) {
		for (var x = lo.x; x <= hi.x; x++) {
			let p = ((vec2i(x, y) % sourceSize) + sourceSize) % sourceSize;
			maximum = max(maximum, textureLoad(weather, vec2u(p)).r);
		}
	}
	return min(1.0, maximum + 1.0 / 255.0);
}

fn scShellHeight(p: vec3f, origin: vec3f) -> f32 {
	let horizontal = p.xz - origin.xz;
	let altitude = p.y + dot(horizontal, horizontal) / (2.0 * 6371000.0);
	return (altitude - 4000.0) / 5200.0;
}

fn scConeLod(footprint: f32, scale: f32) -> f32 {
	return max(0.0, log2(max(footprint * 64.0 / scale, 1.0)));
}

fn scNoise(tex: texture_2d<f32>, uvw: vec3f) -> vec3f {
	let n = 64;
	let p = fract(uvw) * f32(n) - 0.5;
	let i = vec3i(floor(p));
	let f = fract(p);
	var acc = vec3f(0.0);
	for (var z = 0; z <= 1; z++) {
		for (var y = 0; y <= 1; y++) {
			for (var x = 0; x <= 1; x++) {
				let c = vec3i(scImod(i.x + x, n), scImod(i.y + y, n), scImod(i.z + z, n));
				let tile = vec2i(c.z % 8, c.z / 8);
				let px = vec2u(u32(tile.x * n + c.x), u32(tile.y * n + c.y));
				let wx = select(1.0 - f.x, f.x, x == 1);
				let wy = select(1.0 - f.y, f.y, y == 1);
				let wz = select(1.0 - f.z, f.z, z == 1);
				acc += textureLoad(tex, px, 0).rgb * (wx * wy * wz);
			}
		}
	}
	return acc;
}

struct ScCandidate {
	conservative: f32,
	edge: f32,
	top: f32,
	floorMask: f32,
	shellHeight: f32,
	localHeight: f32,
	coverage: f32,
	position: vec3f,
};

fn scCandidate(
	p: vec3f,
	coverage: f32,
	lod: f32,
	origin: vec3f,
	windOffset: vec2f,
	windDir: vec2f,
	evolution: f32,
	noise: texture_2d<f32>,
	weather: texture_2d<f32>,
	weatherSmp: sampler
) -> ScCandidate {
	let h = scShellHeight(p, origin);
	var result: ScCandidate;
	result.conservative = 0.0;
	result.position = vec3f(0.0);
	if (h < -0.1 || h > 1.3 || coverage <= 0.0) { return result; }
	let shapePosition = p - vec3f(windOffset.x, 0.0, windOffset.y)
		- vec3f(windDir.x, 0.0, windDir.y) * (1750.0 * max(h, 0.0) - evolution);
	let weatherValue = textureSampleLevel(weather, weatherSmp, (p.xz - windOffset) / 29000.0, 0.0).r;
	let edge = max(0.095 * exp2(-max(h, 0.0) * 0.0), 0.0001);
	let maximumTop = weatherValue + coverage - 1.0 + 1.34 * 0.69 * coverage;
	if (maximumTop < h - edge) { return result; }
	let required = (1.0 - smoothstep(0.0, 0.13, h)) * 0.54;
	let floorMask = smoothstep(required - 0.1, required, weatherValue);
	if (floorMask <= 0.0) { return result; }
	let baseSample = scNoise(noise, shapePosition / 7500.0);
	let dilation = dot(baseSample, vec3f(0.7, 0.41, 0.23)) * 0.69;
	let top = weatherValue + coverage - 1.0 + dilation * coverage;
	result.conservative = smoothstep(-edge, edge, top - h) * smoothstep(-edge, edge, h) * floorMask;
	result.edge = edge;
	result.top = top;
	result.floorMask = floorMask;
	result.shellHeight = h;
	result.localHeight = clamp(h / max(top, 0.001), 0.0, 1.0);
	result.coverage = coverage;
	result.position = shapePosition;
	return result;
}

fn scErodedDensity(c: ScCandidate, lod: f32, noise: texture_2d<f32>) -> f32 {
	if (c.conservative <= 0.0) { return 0.0; }
	let strength = mix(0.24, 2.15, c.localHeight);
	let maximumErosion = 0.173 * strength * c.coverage;
	if (min(c.top - c.shellHeight, c.shellHeight) >= maximumErosion + c.edge) { return c.floorMask; }
	let s = scNoise(noise, c.position / (7500.0 * 0.13));
	let field = mix(1.0 - s, s, vec3f(1.0));
	let erosion = dot(field, vec3f(0.113, 0.04, 0.02)) * strength * c.coverage;
	return smoothstep(-c.edge, c.edge, c.top - erosion - c.shellHeight)
		* smoothstep(-c.edge, c.edge, c.shellHeight - erosion) * c.floorMask;
}

fn scLightDensity(
	p: vec3f,
	coverage: f32,
	lods: vec2f,
	cheap: bool,
	origin: vec3f,
	windOffset: vec2f,
	windDir: vec2f,
	evolution: f32,
	noise: texture_2d<f32>,
	weather: texture_2d<f32>,
	weatherSmp: sampler
) -> f32 {
	let c = scCandidate(p, coverage, lods.x, origin, windOffset, windDir, evolution, noise, weather, weatherSmp);
	if (cheap) { return c.conservative; }
	return scErodedDensity(c, lods.y, noise);
}

fn scShellRoots(direction: vec3f, altitude: f32, originHeight: f32) -> vec2f {
	let radius = 6371000.0;
	let b = (radius + originHeight) * direction.y;
	let c = (originHeight - altitude) * (2.0 * radius + originHeight + altitude);
	let h = b * b - c;
	if (h < 0.0) { return vec2f(-1.0); }
	let q = -b - select(-sqrt(h), sqrt(h), b >= 0.0);
	let other = c / select(-0.0001, q, abs(q) > 0.0001);
	return vec2f(min(q, other), max(q, other));
}

fn scCloudInterval(dir: vec3f, originHeight: f32) -> vec2f {
	let inner = scShellRoots(dir, 4000.0, originHeight);
	let outer = scShellRoots(dir, 9200.0, originHeight);
	var start = 0.0;
	var end = min(outer.y, 250000.0);
	if (outer.y <= 0.0) { return vec2f(0.0); }
	if (originHeight < 4000.0) { start = max(inner.y, 0.0); }
	else {
		if (originHeight > 9200.0) { start = max(outer.x, 0.0); }
		if (inner.x >= 0.0) { end = min(end, inner.x); }
	}
	let ground = scShellRoots(dir, 0.0, originHeight);
	if (ground.x > 0.0) { end = min(end, ground.x); }
	return vec2f(start, end);
}

fn scCoverageAt(t: f32, coverage: f32) -> f32 {
	return coverage + 0.12 * smoothstep(20000.0, 65000.0, t);
}

fn scHeightAlongRay(t: f32, dir: vec3f, originHeight: f32) -> f32 {
	let altitude = originHeight + dir.y * t + dot(dir.xz, dir.xz) * t * t / (2.0 * 6371000.0);
	return (altitude - 4000.0) / 5200.0;
}

fn scCellIsEmpty(t: f32, end: f32, dir: vec3f, originHeight: f32, maximumWeather: f32, coverage: f32) -> bool {
	let turningPoint = -dir.y * 6371000.0 / max(dot(dir.xz, dir.xz), 0.000001);
	let minHeight = scHeightAlongRay(clamp(turningPoint, t, end), dir, originHeight);
	let maxHeight = max(scHeightAlongRay(t, dir, originHeight), scHeightAlongRay(end, dir, originHeight));
	let cover = scCoverageAt(end, coverage);
	let maximumTop = maximumWeather + cover - 1.0 + 1.34 * 0.69 * cover;
	let edge = max(0.095 * exp2(-max(minHeight, 0.0) * 0.0), 0.0001);
	let required = (1.0 - smoothstep(0.0, 0.13, maxHeight)) * 0.54;
	return maximumTop < minHeight - edge || maximumWeather <= required - 0.1;
}

fn scLightTap(i: u32, L: vec3f) -> vec4f {
	if (i == 0u) { return vec4f(0.0, 0.0, 0.0, 25.0); }
	let fi = f32(i);
	let growth = pow(1.7, fi);
	let mid = 25.0 * ((growth - 1.0) / 0.7 + growth * 0.5);
	let angle = fi * 2.399963;
	let radius = sqrt((fi + 0.5) / 6.0);
	let u = cos(angle) * radius * 0.05 * mid;
	let v = sin(angle) * radius * 0.05 * mid;
	var tx = L.z;
	var ty = 0.0;
	var tz = -L.x;
	if (abs(L.y) > 0.99) { tx = 0.0; ty = -L.z; tz = L.y; }
	let inv = inverseSqrt(max(tx * tx + ty * ty + tz * tz, 1e-8));
	tx *= inv; ty *= inv; tz *= inv;
	let bx = L.y * tz - L.z * ty;
	let by = L.z * tx - L.x * tz;
	let bz = L.x * ty - L.y * tx;
	return vec4f(L * mid + vec3f(tx, ty, tz) * u + vec3f(bx, by, bz) * v, 25.0 * growth);
}

fn scLightLod(i: u32) -> vec2f {
	if (i == 0u) { return vec2f(0.0); }
	let growth = pow(1.7, f32(i));
	let mid = 25.0 * ((growth - 1.0) / 0.7 + growth * 0.5);
	let footprint = mid * 0.1;
	return vec2f(scConeLod(footprint, 7500.0), scConeLod(footprint, 7500.0 * 0.13));
}

fn scLightOpticalDepth(
	p: vec3f,
	coverage: f32,
	lods: vec2f,
	cheap: bool,
	sunDir: vec3f,
	origin: vec3f,
	windOffset: vec2f,
	windDir: vec2f,
	evolution: f32,
	noise: texture_2d<f32>,
	weather: texture_2d<f32>,
	weatherSmp: sampler
) -> f32 {
	var opticalDepth = 0.0;
	for (var i = 1u; i < 8u; i++) {
		if (i >= 6u) { break; }
		let tap = scLightTap(i, sunDir);
		opticalDepth += scLightDensity(p + tap.xyz, coverage, max(lods, scLightLod(i)), cheap, origin, windOffset, windDir, evolution, noise, weather, weatherSmp) * tap.w * 0.019;
		if (opticalDepth >= 32.0) { break; }
	}
	return opticalDepth;
}

fn scLightEnergy(opticalDepth: f32, phase: vec3f) -> f32 {
	let quarter = exp(-opticalDepth * 0.25);
	let halfT = quarter * quarter;
	return dot(vec3f(halfT * halfT, halfT * 0.5, quarter * 0.25), phase);
}

fn scRay(uv: vec2f, right: vec3f, up: vec3f, forward: vec3f, tanY: f32, aspect: f32) -> vec3f {
	let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	return normalize(forward + right * (ndc.x * aspect * tanY) + up * (ndc.y * tanY));
}

fn scMarch(
	origin: vec3f,
	dir: vec3f,
	dither: f32,
	pixelCone: f32,
	stepCone: f32,
	coverage: f32,
	windOffset: vec2f,
	windDir: vec2f,
	evolution: f32,
	sunDir: vec3f,
	night: f32,
	skyIrr: vec3f,
	viewHeight: f32,
	depthOut: ptr<private, f32>,
	noise: texture_2d<f32>,
	weather: texture_2d<f32>,
	weatherSmp: sampler,
	bounds: texture_storage_2d<rgba8unorm, read>,
	sky: texture_storage_2d<rgba16float, read>,
	trans: texture_storage_2d<rgba16float, read>
) -> vec4f {
	var color = vec3f(0.0);
	var transmission = 1.0;
	var weightedDepth = 0.0;
	let interval = scCloudInterval(dir, origin.y);
	if (interval.y > interval.x && coverage > 0.0) {
		let lightCosine = dot(dir, sunDir);
		let phase = scPhases(lightCosine);
		let direct = scDirect(trans, sunDir);
		let zenith = scSky(sky, viewHeight, sunDir, vec3f(0.0, 1.0, 0.0), skyIrr, night);
		let horizon = scSky(sky, viewHeight, sunDir, normalize(vec3f(sunDir.x, 0.03, sunDir.z) + vec3f(1e-5, 0.0, 0.0)), skyIrr, night);
		let bounce = vec3f(0.0091340587, 0.0152085144, 0.0185002201) * direct * max(sunDir.y, 0.0);
		var t = interval.x + dither * 25.0;
		var coarse = true;
		var emptyRun = 0u;
		var accumulatedDepth = 0.0;
		var refineEnd = -1.0;
		var cellEnd = -1.0;
		var cellStart = -1.0;
		var maximumWeather = 1.0;
		var shadowAt = -250000.0;
		var shadowDensity = -1.0;
		var shadowTau = 0.0;
		var skyTau = 0.0;
		var shadowCheap = false;
		let hasDirectLight = max(direct.x, max(direct.y, direct.z)) > 0.00001;
		for (var i = 0u; i < 256u; i++) {
			if (t >= interval.y || transmission < 0.003) { break; }
			if (coarse) {
				if (t >= cellEnd || t < cellStart) {
					let grid = (origin.xz + dir.xz * t - windOffset) / 29000.0 * vec2f(64.0);
					let cell = vec2i(floor(grid));
					let wrapped = ((cell % vec2i(64)) + vec2i(64)) % vec2i(64);
					maximumWeather = textureLoad(bounds, vec2u(wrapped)).r;
					let distance = select(fract(grid), 1.0 - fract(grid), dir.xz >= vec2f(0.0)) / max(abs(dir.xz) * vec2f(64.0) / 29000.0, vec2f(0.0000001));
					cellStart = t;
					cellEnd = min(interval.y, t + max(min(distance.x, distance.y), 0.05));
				}
				if (scCellIsEmpty(t, cellEnd, dir, origin.y, maximumWeather, coverage)) { t = cellEnd + 0.05; continue; }
			}
			let footprint = t * pixelCone;
			let fineStep = max(25.0, t * stepCone * 1.5);
			let p = origin + dir * t;
			let cover = scCoverageAt(t, coverage);
			let baseLod = scConeLod(footprint, 7500.0);
			let c = scCandidate(p, cover, baseLod, origin, windOffset, windDir, evolution, noise, weather, weatherSmp);
			let erosionLod = scConeLod(footprint, 7500.0 * 0.13);
			if (coarse) {
				if (c.conservative > 0.0) {
					if (scErodedDensity(c, erosionLod, noise) > 0.0) {
						refineEnd = t;
						t = max(interval.x, t - fineStep * 4.0) + fineStep;
						t = min(t, refineEnd);
						coarse = false;
						emptyRun = 0u;
						continue;
					}
				}
				t += fineStep * 4.0;
				continue;
			}
			let density = scErodedDensity(c, erosionLod, noise);
			if (density <= 0.0) {
				shadowAt = -250000.0;
				emptyRun++;
				if (t < refineEnd) { t = min(t + fineStep, refineEnd); continue; }
				if (emptyRun >= 4u) { coarse = true; }
				t += select(fineStep, fineStep * 4.0, coarse);
				continue;
			}
			emptyRun = 0u;
			refineEnd = -1.0;
			let sigmaT = density * 0.019;
			let surfaceStep = clamp(0.5 / max(sigmaT, 0.000001), fineStep * 0.15, fineStep);
			let stepLength = min(mix(surfaceStep, fineStep * 3.0, smoothstep(1.0, 3.0, accumulatedDepth)), interval.y - t);
			let height = clamp(c.shellHeight, 0.0, 1.0);
			let baseShadow = mix(1.0, mix(0.15, 1.0, smoothstep(0.0, 0.13, height)), 0.88);
			let powderWeight = 0.7 * (1.0 - smoothstep(0.2, 0.95, lightCosine));
			let powder = mix(1.0, 1.0 - exp(-density * 2.0), powderWeight);
			var energy = 0.0;
			let cheap = (1.0 - transmission) >= 0.5;
			let reuseDistance = min(fineStep * 3.0, max(20.0, 7500.0 * 0.13 * 0.03));
			if (abs(t - shadowAt) >= reuseDistance || abs(density - shadowDensity) > 0.08 || cheap != shadowCheap) {
				let lods = vec2f(baseLod, erosionLod);
				if (hasDirectLight) {
					shadowTau = scLightOpticalDepth(p, cover, lods, cheap, sunDir, origin, windOffset, windDir, evolution, noise, weather, weatherSmp);
				}
				let skyLods = lods + vec2f(1.0);
				skyTau = (scLightDensity(p + vec3f(0.0, 125.0, 0.0), cover, skyLods, true, origin, windOffset, windDir, evolution, noise, weather, weatherSmp) * 250.0
					+ scLightDensity(p + vec3f(0.0, 600.0, 0.0), cover, skyLods, true, origin, windOffset, windDir, evolution, noise, weather, weatherSmp) * 700.0) * 0.019;
				shadowAt = t;
				shadowDensity = density;
				shadowCheap = cheap;
			}
			if (hasDirectLight) { energy = scLightEnergy(shadowTau + sigmaT * 25.0, phase); }
			let skyVisibility = 0.2 + 0.8 / (1.0 + (skyTau + sigmaT * 25.0) * 0.35);
			let ambient = (mix(mix(zenith, horizon, 0.55), zenith, sqrt(height)) * skyVisibility + bounce * (1.0 - height)) * 0.7 * (1.0 + 0.99);
			let light = (direct * energy * powder * baseShadow + ambient) * 1.0;
			let stepT = exp(-sigmaT * stepLength);
			let stepAlpha = transmission * (1.0 - stepT);
			color += stepAlpha * light;
			weightedDepth += stepAlpha * t;
			transmission *= stepT;
			accumulatedDepth += sigmaT * stepLength;
			t += stepLength;
		}
	}
	let alpha = 1.0 - transmission;
	var depth = 250000.0;
	if (alpha > 0.0) {
		depth = weightedDepth / alpha;
		let skyColor = scSky(sky, viewHeight, sunDir, dir, skyIrr, night);
		let ap = exp(-depth / 30000.0);
		color = color * ap + skyColor * alpha * (1.0 - ap);
		let melt = smoothstep(25000.0, 45000.0, depth);
		color = mix(color, skyColor * alpha, melt);
	}
	*depthOut = depth;
	return vec4f(color, alpha);
}

fn scHistoryAt(uv: vec2f, size: vec2f, tex: texture_2d<f32>, smp: sampler) -> vec4f {
	let position = uv * size;
	let center = floor(position - 0.5) + 0.5;
	let fr = position - center;
	let w0 = fr * (fr * (-0.5 * fr + 1.0) - 0.5);
	let w1 = fr * fr * (1.5 * fr - 2.5) + 1.0;
	let w2 = fr * (fr * (-1.5 * fr + 2.0) + 0.5);
	let w3 = fr * fr * (0.5 * fr - 0.5);
	let w12 = w1 + w2;
	let uv0 = (center - 1.0) / size;
	let uv3 = (center + 2.0) / size;
	let uv12 = (center + w2 / w12) / size;
	let weights = vec4f(w12.x * w0.y, w0.x * w12.y, w12.x * w12.y, w3.x * w12.y);
	let bottomWeight = w12.x * w3.y;
	let value = textureSampleLevel(tex, smp, vec2f(uv12.x, uv0.y), 0.0) * weights.x
		+ textureSampleLevel(tex, smp, vec2f(uv0.x, uv12.y), 0.0) * weights.y
		+ textureSampleLevel(tex, smp, uv12, 0.0) * weights.z
		+ textureSampleLevel(tex, smp, vec2f(uv3.x, uv12.y), 0.0) * weights.w
		+ textureSampleLevel(tex, smp, vec2f(uv12.x, uv3.y), 0.0) * bottomWeight;
	return value / (dot(weights, vec4f(1.0)) + bottomWeight);
}

fn scFreshWeight(color: vec4f, depth: f32, historyWeight: f32, evolve: f32) -> f32 {
	let retain = select(mix(0.25, 0.65, smoothstep(8.0, 30.0, depth)), 0.25, color.a < 0.0001);
	return min(historyWeight, retain) * (1.0 - evolve);
}

fn scProject(world: vec3f, prevPos: vec3f, prevRight: vec3f, prevUp: vec3f, prevFwd: vec3f, prevTanY: f32, prevAspect: f32) -> vec3f {
	let v = world - prevPos;
	let z = dot(v, prevFwd);
	let denom = max(z, 0.001) * prevTanY;
	return vec3f(vec2f(dot(v, prevRight) / (denom * prevAspect), -dot(v, prevUp) / denom) * 0.5 + 0.5, z);
}
` );

/** 写天气图的一个纹素。只含红通道覆盖。 */
export const skyproWeather = wgslFn( /* wgsl */`
fn skyproWeather(uv: vec2f) -> vec4f {
	return scWeather(uv);
}
`, [ skyproLib as never ] );

/** 64² 边界格里的最大天气值，空域跳步用。 */
export const skyproBounds = wgslFn( /* wgsl */`
fn skyproBounds(id: vec2u, weather: texture_storage_2d<rgba8unorm, read>) -> f32 {
	return scBoundsAt(id, weather);
}
`, [ skyproLib as never ] );

/**
 * 追踪四分之一格子里的一个槽。
 *
 * 返回 `vec4(内散射, 不透明度)`，距离（米）写入 `depthOut`。
 */
export const skyproTrace = wgslFn( /* wgsl */`
fn skyproTrace(
	uv: vec2f,
	dither: f32,
	pixelCone: f32,
	stepCone: f32,
	origin: vec3f,
	right: vec3f,
	up: vec3f,
	forward: vec3f,
	tanY: f32,
	aspect: f32,
	coverage: f32,
	windOffset: vec2f,
	windDir: vec2f,
	evolution: f32,
	sunDir: vec3f,
	night: f32,
	skyIrr: vec3f,
	viewHeight: f32,
	depthOut: ptr<private, f32>,
	noise: texture_2d<f32>,
	weather: texture_2d<f32>,
	weatherSmp: sampler,
	bounds: texture_storage_2d<rgba8unorm, read>,
	sky: texture_storage_2d<rgba16float, read>,
	trans: texture_storage_2d<rgba16float, read>
) -> vec4f {
	let dir = scRay(uv, right, up, forward, tanY, aspect);
	return scMarch(origin, dir, dither, pixelCone, stepCone, coverage, windOffset, windDir, evolution, sunDir, night, skyIrr, viewHeight, depthOut, noise, weather, weatherSmp, bounds, sky, trans);
}
`, [ skyproLib as never ] );

/**
 * 把本帧的格子样本写进半分辨率历史。
 *
 * `outColor` 是 `vec4(内散射, 不透明度)`，`outMeta.x` 是当前深度（千米），`y` 是重投影用的深度。
 */
export const skyproResolve = wgslFn( /* wgsl */`
fn skyproResolve(
	id: vec2u,
	historySize: vec2f,
	sourceSize: vec2f,
	slot: vec2f,
	origin: vec3f,
	right: vec3f,
	up: vec3f,
	forward: vec3f,
	tanY: f32,
	aspect: f32,
	prevPos: vec3f,
	prevRight: vec3f,
	prevUp: vec3f,
	prevFwd: vec3f,
	prevTanY: f32,
	prevAspect: f32,
	windDelta: vec2f,
	historyWeight: f32,
	evolve: f32,
	historyValid: f32,
	outMeta: ptr<private, vec4f>,
	curColor: texture_storage_2d<rgba16float, read>,
	curMeta: texture_storage_2d<rgba16float, read>,
	prevColor: texture_2d<f32>,
	prevSmp: sampler,
	prevMeta: texture_storage_2d<rgba16float, read>
) -> vec4f {
	let size = vec2i(historySize);
	let srcSize = vec2i(sourceSize) - vec2i(1);
	let uv = (vec2f(id) + 0.5) / historySize;
	let fresh = all(id % vec2u(4u) == vec2u(u32(slot.x), u32(slot.y)));
	let source = (vec2f(id) - slot) / 4.0;
	let center = clamp(vec2i(round(source)), vec2i(0), srcSize);
	let metadata = textureLoad(curMeta, vec2u(center));
	let central = textureLoad(curColor, vec2u(center));
	let currentDepth = max(metadata.x, 0.001);
	let stored = textureLoad(prevMeta, id);
	var color = vec4f(0.0);
	var weightSum = 0.0;
	let base = vec2i(floor(source));
	let fraction = fract(source);
	for (var y = 0; y < 2; y++) {
		for (var x = 0; x < 2; x++) {
			let pos = clamp(base + vec2i(x, y), vec2i(0), srcSize);
			let tap = textureLoad(curColor, vec2u(pos));
			let depth = textureLoad(curMeta, vec2u(pos)).x;
			let bilinear = select(1.0 - fraction.x, fraction.x, x == 1) * select(1.0 - fraction.y, fraction.y, y == 1);
			let weight = bilinear * exp(-abs(depth - metadata.x) / max(0.1, metadata.x * 0.1) - abs(tap.a - central.a) * 4.0);
			color += tap * weight;
			weightSum += weight;
		}
	}
	color = select(central, color / max(weightSum, 0.00001), weightSum > 0.00001);
	if (fresh) { color = central; }
	if (historyValid < 0.5) {
		*outMeta = vec4f(currentDepth, currentDepth, metadata.z, 0.0);
		return vec4f(max(color.rgb, vec3f(0.0)), clamp(color.a, 0.0, 1.0));
	}
	var low = color;
	var high = color;
	var reprojectionDepth = select(250.0, stored.y, stored.y >= 0.001);
	for (var y = -1; y <= 1; y++) {
		for (var x = -1; x <= 1; x++) {
			let tap = textureLoad(curColor, vec2u(clamp(center + vec2i(x, y), vec2i(0), srcSize)));
			low = min(low, tap);
			high = max(high, tap);
			let neighbor = clamp(vec2i(id) + vec2i(x, y), vec2i(0), size - vec2i(1));
			let depth = textureLoad(prevMeta, vec2u(neighbor)).y;
			if (depth >= 0.001) { reprojectionDepth = min(reprojectionDepth, depth); }
		}
	}
	if (stored.y < 0.001) { reprojectionDepth = min(reprojectionDepth, currentDepth); }
	var world = origin + scRay(uv, right, up, forward, tanY, aspect) * (reprojectionDepth * 1000.0);
	world -= vec3f(windDelta.x, 0.0, windDelta.y);
	let previous = scProject(world, prevPos, prevRight, prevUp, prevFwd, prevTanY, prevAspect);
	let previousPixel = clamp(vec2i(previous.xy * historySize), vec2i(0), size - vec2i(1));
	let oldMeta = textureLoad(prevMeta, vec2u(previousPixel));
	let valid = previous.z > 0.0 && all(previous.xy >= vec2f(0.0)) && all(previous.xy <= vec2f(1.0)) && oldMeta.y >= 0.001;
	var carriedDepth = currentDepth;
	var historyFraction = 0.0;
	if (valid) {
		let history = clamp(scHistoryAt(previous.xy, historySize, prevColor, prevSmp), low, high);
		let motionPixels = length((previous.xy - uv) * historySize);
		historyFraction = select(1.0, scFreshWeight(central, currentDepth, historyWeight, evolve), fresh);
		color = mix(color, history, historyFraction);
		let expectedDepth = length(world - prevPos) * 0.001;
		if (!fresh && (motionPixels <= 0.5 || abs(oldMeta.y - expectedDepth) < max(0.001, expectedDepth * 0.15))) {
			carriedDepth = oldMeta.y;
		}
	}
	*outMeta = vec4f(currentDepth, carriedDepth, metadata.z, historyFraction);
	return vec4f(max(color.rgb, vec3f(0.0)), clamp(color.a, 0.0, 1.0));
}
`, [ skyproLib as never ] );
