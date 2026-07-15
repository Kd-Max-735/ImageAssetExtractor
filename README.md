# Image Asset Extractor

从一张 PNG/JPG/JPEG/WebP 图片中自动定位多个素材，分别输出透明 PNG，并生成 JSON、CSV、标注图和 ZIP。内置同源 Web 前端可完成上传、检测、纠错、预览和导出。

## 支持范围

- 可靠路径：有效透明背景、白底、单一纯色背景；近似纯色和渐变使用边界平面局部背景模型。
- 基础能力：预检、背景分类、Lab 色差、软/硬蒙版、自适应形态学、完整连通域特征、评分聚类、规则网格/空白投影、投影谷与距离变换/分水岭粘连拆分、排序、去背景色污染、留白及批量导出。
- 人工纠错：合并、自动/分割线/矩形拆分、删除、bbox、软硬画笔/羽化、撤销/重做、排序、重命名；每次操作执行 revision 冲突检查并重建全部导出物。
- 实验状态：近似纯色、渐变、复杂背景、严重粘连。
- RMBG 仅用于候选 ROI 的软 Alpha 优化；SAM 3.1 仅作为低置信度/提示分割适配器。二者不可用时不影响 OpenCV 基础路径。
- 不支持：SVG、位图转矢量、视频、遮挡补全、三维恢复和超分辨率。

## 安装

要求 Python 3.10+。建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## 环境变量

复制 `.env.example` 后按需设置：

| 变量 | 说明 |
|---|---|
| `IMAGE_ASSET_OUTPUT_DIR` | API 任务输出根目录，默认 `output` |
| `EXTRACTION_MAX_WORKERS` | 同时执行的提取任务数；8 GB 双模型部署建议为 `1` |
| `ENABLE_RMBG` | 是否启用 RMBG 能力检测 |
| `RMBG_MODEL_PATH` | 外部 RMBG 模型/项目路径，不复制权重 |
| `RMBG_SERVICE_URL` | 独立 RMBG HTTP 服务地址 |
| `RMBG_REMOVE_BG_PATH` | RMBG 推理端点，默认 `/remove-bg` |
| `ENABLE_SAM31` | 是否启用 SAM 3.1，默认关闭 |
| `SAM31_MODEL_PATH` | 外部 SAM 3.1 路径 |
| `SAM31_SERVICE_URL` | 独立 SAM 3.1 服务地址 |
| `SAM31_SEGMENT_PATH` | SAM 3.1 分割端点，默认 `/segment` |

不要把 RMBG 与 SAM 3.1 强行安装到同一 Python 环境。基础包不会导入二者依赖，也不会修改 `D:\RMBG` 或 `D:\sam3.1`。

## CLI

```powershell
python -m image_asset_extractor.cli extract input.png --output output
```

常用参数：

```text
--background-type auto
--asset-mode icon
--shadow-mode auto
--text-mode auto
--min-area-ratio 0.0001
--merge-distance-ratio 0.01
--padding-ratio 0.05
--background-tolerance 15
--output-prefix icon
--number-digits 3
--output-canvas tight
--canvas-size 512
--edge-feather 0
--split-strength 0.5
--include-zip / --no-include-zip
--use-rmbg
--use-sam31
```

成功退出码为 `0`；失败为非零，并向 stderr 输出 `{code,message,details}` JSON。为避免覆盖已有结果，如果目标目录已有 `result/metadata.json`，命令会返回 `OUTPUT_EXISTS`。

## API

启动：

```powershell
uvicorn image_asset_extractor.api.app:app --host 0.0.0.0 --port 8000
```

交互文档位于 `http://localhost:8000/docs`。核心接口：

```text
GET    /health
GET    /capabilities
POST   /api/asset-extraction/tasks
GET    /api/asset-extraction/tasks/{task_id}
GET    /api/asset-extraction/tasks/{task_id}/assets
GET    /api/asset-extraction/tasks/{task_id}/source
GET    /api/asset-extraction/tasks/{task_id}/assets/{asset_id}/preview
GET    /api/asset-extraction/tasks/{task_id}/assets/{asset_id}/download
POST   /api/asset-extraction/tasks/{task_id}/merge
POST   /api/asset-extraction/tasks/{task_id}/split
POST   /api/asset-extraction/tasks/{task_id}/assets/{asset_id}/delete
PUT    /api/asset-extraction/tasks/{task_id}/assets/{asset_id}/bbox
PUT    /api/asset-extraction/tasks/{task_id}/assets/{asset_id}/mask
POST   /api/asset-extraction/tasks/{task_id}/sort
PUT    /api/asset-extraction/tasks/{task_id}/assets/{asset_id}/name
POST   /api/asset-extraction/tasks/{task_id}/undo
POST   /api/asset-extraction/tasks/{task_id}/redo
POST   /api/asset-extraction/tasks/{task_id}/export
GET    /api/asset-extraction/tasks/{task_id}/export/download
DELETE /api/asset-extraction/tasks/{task_id}
POST   /api/asset-extraction/tasks/{task_id}/retry
```

上传示例：

```powershell
curl.exe -X POST http://localhost:8000/api/asset-extraction/tasks `
  -F "file=@input.png" `
  -F "backgroundType=auto" `
  -F "paddingRatio=0.05"
```

创建接口返回 `202` 和 `task_id`；轮询任务状态到 `completed` 后获取素材或导出。任务返回实际 `progress` 和 `stage`。长耗时提取在线程池中执行，ZIP/重导出位于同步工作线程，不阻塞事件循环。所有错误均包含 `code`、`message` 和 `details`。

所有编辑请求必须提交当前 `revision`；冲突返回 `REVISION_CONFLICT`（HTTP 409）。编辑成功后 revision 增加，并在同一次受控编辑提交中重建当前任务的 PNG、JSON、CSV、标注图和（启用时）ZIP。`includeZip=false` 不生成 ZIP；之后可调用导出接口并传 `{"includeZip":true}` 显式生成。

## Web 前端

启动 API 后访问 `http://localhost:8000/`。前端使用原生 HTML/CSS/JavaScript，无 Node 运行时依赖；页面直接调用上述 API，支持原图框选、低置信度高亮、缩放、分割线、蒙版画笔、黑/白/透明棋盘预览、编辑历史及下载。

## 输出结构

CLI 的 `--output` 是输出根目录。API 自动使用独立的 `output/{task_id}`：

```text
output/
├── icons/
│   ├── icon_001.png
│   └── icon_002.png
└── result/
    ├── metadata.json
    ├── metadata.csv
    ├── annotated.png
    └── assets.zip
```

ZIP 保留 `icons/` 和 `result/` 路径，但不包含 ZIP 自身。所有坐标使用原图像素。

`metadata.json` 顶层包含 `sourceFile`、`sourceWidth`、`sourceHeight`、`backgroundType`、`backgroundConfidence`、`backgroundStatus`、`backgroundReasons`、`assetCount` 和 `assets`。每个素材包含：

```text
id, file, index, x, y, width, height, maskArea, confidence,
confidenceReasons, regionIds, flags, status
```

CSV 每行对应一个 PNG，编号与 JSON 和文件名一致。

## RMBG 与 SAM 3.1

`GET /capabilities` 分别报告 OpenCV、RMBG 和 SAM 3.1 的 `enabled`、`available`、运行模式及不可用原因。服务模式会执行 `/health`，没有健康端点的 RMBG 服务则探测 OpenAPI，不再仅凭 URL 判为可用。RMBG 对候选 ROI 生成软 Alpha；SAM 3.1 接受框提示，并通过尺寸、类型、数值、面积比、IoU 和边界门控避免低质量掩码破坏可靠结果。每个候选独立回退，并在 `confidenceReasons` 记录模型、拒绝原因和置信度变化。两个模型始终在各自隔离环境中加载。

## 测试

测试图片由代码生成，不依赖外部下载：

```powershell
python -m pytest
```

覆盖透明/白底/纯色/近似纯色/渐变/复杂背景、JPG 噪声、低对比同色主体、规则网格、不规则排列、多组件、近距离同色对象、轻微粘连、超过 100 个素材、软 Alpha、方形画布、EXIF、格式伪装、空图、资源/候选上限、全部纠错操作、revision 冲突、JSON/CSV/标注图/ZIP 一致性、CLI、异步 API、前端静态冒烟以及模型可用/不可用路径。

本机合成集实测（OpenCV 路径、无 ZIP/模型）：透明底 2000×2000、25 个素材约 0.579 秒，进程 RSS 约 67 MiB；透明底 5000×5000、36 个素材约 4.545 秒，约 154 MiB；渐变底 5000×5000、16 个素材约 4.761 秒，约 228 MiB。该数据只代表当前机器和合成用例，不代表生产图片或准确率。

## Docker + RMBG + SAM 3.1

本机验证架构为：Docker Desktop/WSL2 运行 CPU 业务 API；RMBG 和 SAM 3.1 使用已经配置好的两个 Windows GPU 环境，通过 `host.docker.internal` 提供独立服务。这样不会复制权重，也不会混装 PyTorch 2.6/CUDA 12.4 与 PyTorch 2.7/CUDA 12.6。

当前机器完成首次构建后，可用一个命令启动完整栈：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_stack.ps1
```

需要强制重建 API 镜像时增加 `-Build`。也可以分别启动模型服务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_rmbg_service.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_sam31_service.ps1
```

再启动业务 API：

```powershell
Copy-Item .env.docker.example .env
docker compose up -d --build api
docker compose ps
```

默认地址：

```text
项目 API / 文档: http://localhost:8000 / http://localhost:8000/docs
RMBG:            http://localhost:8001
SAM 3.1:         http://localhost:8002/health
```

验证服务与能力：

```powershell
Invoke-RestMethod http://localhost:8002/health
Invoke-RestMethod http://localhost:8000/capabilities
docker compose exec -T api python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:8001/docs').status); print(urllib.request.urlopen('http://host.docker.internal:8002/health').status)"
```

上传时显式启用两个模型：

```powershell
curl.exe -X POST http://localhost:8000/api/asset-extraction/tasks `
  -F "file=@input.png" `
  -F "useRmbg=true" `
  -F "useSam31=true"
```

RMBG 的纯容器方案保留为可选 profile，并只读挂载 `D:/RMBG/RMBG/RMBG20`：

```powershell
docker compose --profile containerized-rmbg up -d --build rmbg
```

日志和停止命令不会删除容器、镜像、卷或输出文件：

```powershell
docker compose logs -f api
docker compose stop
```

RTX 4070 Laptop 8 GB 已验证可让两个模型同时常驻并依次推理，但空闲显存很少；生产并发应限制为 1，避免并行模型请求导致 OOM。

## 常见问题

- `IMAGE_DECODE_FAILED`：文件不是可解码的受支持图片。
- `FORMAT_MISMATCH`：扩展名与实际解码格式不一致。
- `EMPTY_IMAGE` / `FULLY_TRANSPARENT`：没有可提取前景。
- `IMAGE_TOO_LARGE` / `FILE_TOO_LARGE`：超过尺寸或资源限制。
- `NO_ASSETS`：阈值下没有有效候选，可调整背景类型或容差。
- `OUTPUT_EXISTS`：目标目录已有结果；换一个输出目录，系统不会覆盖。
- `RMBG_UNAVAILABLE` / `SAM31_UNAVAILABLE`：检查启用开关、路径或独立服务地址。

白色物体叠白底、复杂照片背景和严重遮挡本质上信息不足，会返回实验/低置信度结果，需模型或人工提示。当前没有带标准答案的标注数据集，因此 95%/90% 准确率尚未验证；不得从合成回归测试推导生产准确率。
