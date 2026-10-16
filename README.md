# hillaire-sky-webgpu

用 three.js `0.186` WebGPU 做的物理天空演示。大气按 Hillaire EGSR 2020 预计算三张查找表，再叠体积云、场地、后期。

需要能跑 WebGPU 的浏览器（Chrome / Edge 一类）。

## 运行

```bash
pnpm install
pnpm dev
```

打开 http://localhost:5173/ 。`?demo=1` 会跑约 120 秒的轨道时间轴，结束后把控制交回。

```bash
pnpm build
pnpm preview
```

## 操作

- 默认轨道相机，拖旋转、滚轮拉近。
- `V` 在轨道和第一人称之间切。第一人称要点画布锁指针，`Esc` 解锁。
- 右侧面板：天空（时刻、云量）、相机、效果（曝光、泛光、炫光、调色）。

## 画面怎么串

每帧先更新天空（查找表、积云、太阳光），再走后期：

深度预通道（单采样，给 GTAO）→ 场景（MSAA）→ 泛光 → 雾 → 炫光 → 自动曝光 → 调色 → 显示变换（ACES、sRGB、抖动）。

炫光可见度和曝光适应在 `pipeline.render()` 之后 compute，晚一帧生效。环境贴图由 `SkyProbe` 单独烤，不含太阳盘和视锥云。

## 目录

| 路径 | 内容 |
|---|---|
| `src/sky/` | 大气查找表、天空盒、积云场 |
| `src/env/` | 天空环境探针 |
| `src/bloom/` `src/ssf/` `src/glare/` `src/exposure/` `src/grade/` | 后期 |
| `src/trees.ts` | 实例化树林 |
