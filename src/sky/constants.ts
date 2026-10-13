import { Color, Vector3 } from 'three/webgpu';

/**
 * 大气与天空使用的物理常数和场景单位。
 *
 * 数值对齐 Hillaire EGSR 2020 第 4 节。
 * 距离在大气计算里用千米；云和相机高度用米。米氏消光取 Bruneton / Unreal
 * 参考实现的 `4.440e-3 km^-1`（散射 `3.996e-3`，吸收约 `0.444e-3`），
 * 不按论文表 1 把 `4.40e-6 m^-1` 整列当成吸收。
 */

/** 地面半径，单位 km。论文中的 `Rground`。 */
export const GROUND_RADIUS_KM = 6360;

/** 大气顶半径，单位 km。论文中的 `Rtop`，壳层厚 100 km。 */
export const ATMOSPHERE_RADIUS_KM = 6460;

/**
 * 大气顶之外的太阳照度，三个通道相同。
 *
 * 这是场景单位，不是物理勒克斯。取 `11` 以免 fp16 缓冲溢出。
 */
export const SUN_ILLUMINANCE = 11;

/**
 * 太阳盘角半径，单位弧度。
 *
 * 物理视半径约 `0.004675` rad（0.27°）。系数 `1.15`
 * 让低分辨率天空里的日盘稍大一点。
 */
export const SUN_ANGULAR_RADIUS = 0.004675 * 1.15;

/**
 * 太阳盘辐射强度，已经除过立体角后的场景单位上限。
 *
 * 物理值约为照度除以立体角（约 `1.6e5`）。钳到 `2500` 以适配 fp16。
 */
export const SUN_DISK_INTENSITY = 2500;

/** 透射查找表宽度，对应天顶角方向。 */
export const TRANSMITTANCE_WIDTH = 256;

/** 透射查找表高度，对应地心距离。 */
export const TRANSMITTANCE_HEIGHT = 64;

/** 多次散射查找表边长。两个轴分别是太阳天顶角余弦和归一化高度。 */
export const MULTI_SCATTER_SIZE = 32;

/** 天空视图查找表宽度，对应相对太阳的方位。 */
export const SKY_VIEW_WIDTH = 192;

/** 天空视图查找表高度，对应观察天顶角。地平线在 `v = 0.5`。 */
export const SKY_VIEW_HEIGHT = 108;

/**
 * 默认地面反照率。
 *
 * 论文示例用均匀 `0.3`。这里用偏暗海面值，
 * 多次散射因此比论文图更暗。
 */
export const DEFAULT_GROUND_ALBEDO = new Color( 0.06, 0.08, 0.1 );

/** 构造时的太阳方向：东偏南、高度角约 37°。调用方可再 `.copy()`。 */
export const DEFAULT_SUN_DIRECTION = new Vector3( 0.3, 0.6, - 0.7 ).normalize();

/** 默认云量，范围 `0..1`。SkyPro “Partly cloudy” 是 `0.49`。 */
export const DEFAULT_COVERAGE = 0.49;
