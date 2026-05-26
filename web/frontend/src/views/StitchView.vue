<template>
  <div class="stitch-view">
    <!-- 阶段1: 网格预览 -->
    <template v-if="stage === 'preview'">
      <div class="stitch-toolbar">
        <div class="toolbar-left">
          <el-button @click="selectFiles">
            <el-icon><Upload /></el-icon> 选择图片
          </el-button>
          <el-button @click="selectFolder">
            <el-icon><FolderOpened /></el-icon> 选择文件夹
          </el-button>
          <input ref="fileInput" type="file" multiple accept="image/*" style="display:none" @change="onFilesSelected" />
          <input ref="folderInput" type="file" webkitdirectory style="display:none" @change="onFilesSelected" />
        </div>
        <div class="toolbar-right">
          <el-input-number v-model="overlap" :min="0" :max="200" size="small" style="width:120px" />
          <span style="font-size:13px;color:#606266">重叠(px)</span>
          <el-button type="primary" :disabled="tiles.length === 0" :loading="stitching" @click="handleStitchAndDetect">
            拼接并检测
          </el-button>
        </div>
      </div>

      <div class="stitch-body">
        <div v-if="tiles.length === 0" class="stitch-empty">
          <el-icon :size="48" color="#c0c4cc"><Grid /></el-icon>
          <span>选择图片开始拼接（文件名格式: 名称_行_列.ext）</span>
        </div>

        <template v-else>
          <div class="grid-info">
            已解析 <b>{{ tiles.length }}</b> 张图片，网格: <b>{{ gridRows }}</b> 行 × <b>{{ gridCols }}</b> 列
            <span v-if="parseErrors.length" style="color:#e6a23c; margin-left:12px">
              {{ parseErrors.length }} 张文件名不符合格式已忽略
            </span>
            <span style="margin-left:12px; color:#909399">
              预计尺寸: {{ estimatedWidth }} × {{ estimatedHeight }}
            </span>
          </div>

          <div ref="gridRef" class="grid-container" :style="gridStyle">
            <div
              v-for="(cell, idx) in flatGrid"
              :key="idx"
              class="grid-cell"
            >
              <div v-if="cell" class="cell-thumb" :style="{ backgroundImage: `url(${cell.thumbUrl})` }"></div>
              <div v-if="cell" class="cell-label">{{ cell.name }}_{{ cell.row }}_{{ cell.col }}</div>
              <div v-else class="cell-empty">空</div>
            </div>
          </div>
        </template>
      </div>
    </template>

    <!-- 阶段2: 检测结果 -->
    <template v-if="stage === 'result'">
      <div class="stitch-toolbar">
        <div class="toolbar-left">
          <el-button @click="backToPreview">
            <el-icon><Back /></el-icon> 返回
          </el-button>
          <el-select v-model="currentModel" size="small" style="width:160px">
            <el-option v-for="m in models" :key="m.id" :label="m.name" :value="m.id" />
          </el-select>
          <el-radio-group v-model="renderMode" size="small">
            <el-radio-button value="contour">轮廓</el-radio-button>
            <el-radio-button value="pixel">像素</el-radio-button>
          </el-radio-group>
        </div>
        <div class="toolbar-right">
          <el-button size="small" @click="zoomFit">适应窗口</el-button>
          <el-button size="small" @click="zoomReset">100%</el-button>
          <span class="zoom-label">{{ Math.round(scale * 100) }}%</span>
        </div>
      </div>

      <div class="result-body">
        <div class="viewport" ref="viewportRef"
          @wheel.prevent="onWheel"
          @mousedown="onPanStart"
          @mousemove="onPanMove"
          @mouseup="onPanEnd"
          @mouseleave="onPanEnd"
        >
          <div class="transform-layer" :style="transformStyle">
            <img v-if="result" :src="result.image.path" class="stitch-image" draggable="false" @load="onImageLoad" />
            <canvas ref="overlayCanvas" class="stitch-canvas"></canvas>
          </div>
        </div>

        <ResultPanel
          v-if="result"
          :instances="result.instances"
          :summary="result.summary"
          :highlighted-id="highlightedId"
          @hover-instance="highlightedId = $event"
        />
      </div>
    </template>

    <div v-if="stitching" class="stitching-overlay">
      <div class="spinner"></div>
      <div class="text">正在拼接并检测...</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, onBeforeUnmount, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Upload, FolderOpened, Grid, Back } from '@element-plus/icons-vue'
import Sortable from 'sortablejs'
import { Swap } from 'sortablejs/modular/sortable.core.esm'
import ResultPanel from '../components/ResultPanel.vue'
import { getModels, stitchAndDetect } from '../api'

Sortable.mount(new Swap())

// ---- 通用状态 ----
const stage = ref('preview')
const models = ref([])
const currentModel = ref('segformer-b2')
const renderMode = ref('contour')
const stitching = ref(false)
const result = ref(null)
const highlightedId = ref(null)

// ---- 阶段1: 网格预览 ----
const fileInput = ref(null)
const folderInput = ref(null)
const tiles = ref([])        // [{file, name, row, col, thumbUrl}]
const parseErrors = ref([])
const overlap = ref(0)
const gridRef = ref(null)
let sortableInstance = null

// 网格布局（二维数组，gridData[row][col] = tile | null）
const gridData = ref([])

const gridRows = computed(() => gridData.value.length)
const gridCols = computed(() => gridData.value[0]?.length || 0)

const flatGrid = computed(() => {
  const flat = []
  for (const row of gridData.value) {
    for (const cell of row) {
      flat.push(cell)
    }
  }
  return flat
})

const gridStyle = computed(() => ({
  display: 'grid',
  gridTemplateColumns: `repeat(${gridCols.value}, 1fr)`,
  gap: '4px',
  maxWidth: '100%',
}))

const estimatedWidth = computed(() => {
  if (tiles.value.length === 0) return 0
  const sample = tiles.value[0]
  // 从 thumbUrl 无法获取真实尺寸，用缩略图估算
  return gridCols.value * 512
})

const estimatedHeight = computed(() => {
  if (tiles.value.length === 0) return 0
  return gridRows.value * 512
})

function selectFiles() {
  fileInput.value?.click()
}

function selectFolder() {
  folderInput.value?.click()
}

function onFilesSelected(e) {
  const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/'))
  if (files.length === 0) {
    ElMessage.warning('未选择图片文件')
    return
  }
  parseAndBuildGrid(files)
  e.target.value = ''
}

function parseAndBuildGrid(files) {
  const parsed = []
  const errors = []

  for (const file of files) {
    const info = parseFilename(file.name)
    if (info) {
      parsed.push({
        file,
        name: info.name,
        row: info.row,
        col: info.col,
        thumbUrl: URL.createObjectURL(file),
      })
    } else {
      errors.push(file.name)
    }
  }

  if (parsed.length === 0) {
    ElMessage.error('所有文件名都不符合格式（名称_行_列.ext）')
    return
  }

  // 按行列排序
  parsed.sort((a, b) => a.row - b.row || a.col - b.col)

  tiles.value = parsed
  parseErrors.value = errors

  buildGrid()
}

function parseFilename(filename) {
  const stem = filename.replace(/\.[^.]+$/, '')
  const parts = stem.split('_')
  if (parts.length < 3) return null
  try {
    const row = parseInt(parts[parts.length - 2])
    const col = parseInt(parts[parts.length - 1])
    if (isNaN(row) || isNaN(col)) return null
    const name = parts.slice(0, -2).join('_')
    return { name, row, col }
  } catch {
    return null
  }
}

function buildGrid() {
  const maxRow = Math.max(...tiles.value.map(t => t.row))
  const maxCol = Math.max(...tiles.value.map(t => t.col))

  // 初始化空网格
  const grid = []
  for (let r = 0; r <= maxRow; r++) {
    grid.push(new Array(maxCol + 1).fill(null))
  }

  for (const tile of tiles.value) {
    grid[tile.row][tile.col] = tile
  }

  gridData.value = grid
  nextTick(initSortable)
}

function initSortable() {
  if (sortableInstance) {
    sortableInstance.destroy()
    sortableInstance = null
  }
  if (!gridRef.value) return

  sortableInstance = Sortable.create(gridRef.value, {
    swap: true,
    swapClass: 'grid-swap-highlight',
    ghostClass: 'grid-ghost',
    forceFallback: true,
    fallbackClass: 'grid-drag-chosen',
    fallbackOnBody: true,
    fallbackTolerance: 3,
    animation: 150,
    onEnd(evt) {
      if (evt.oldIndex === evt.newIndex) return
      const cols = gridCols.value
      const srcRow = Math.floor(evt.oldIndex / cols)
      const srcCol = evt.oldIndex % cols
      const tgtRow = Math.floor(evt.newIndex / cols)
      const tgtCol = evt.newIndex % cols

      const newGrid = gridData.value.map(r => [...r])
      const tmp = newGrid[srcRow][srcCol]
      newGrid[srcRow][srcCol] = newGrid[tgtRow][tgtCol]
      newGrid[tgtRow][tgtCol] = tmp
      gridData.value = newGrid
      nextTick(initSortable)
    },
  })
}

onBeforeUnmount(() => {
  if (sortableInstance) {
    sortableInstance.destroy()
    sortableInstance = null
  }
})

// 拼接并检测
async function handleStitchAndDetect() {
  stitching.value = true

  // 按网格顺序收集文件
  const orderedFiles = []
  const order = []
  let idx = 0
  for (let r = 0; r < gridRows.value; r++) {
    for (let c = 0; c < gridCols.value; c++) {
      const cell = gridData.value[r][c]
      if (cell) {
        orderedFiles.push(cell.file)
        order.push(idx)
      }
      idx++
    }
  }

  const formData = new FormData()
  for (const f of orderedFiles) {
    formData.append('files', f)
  }
  formData.append('grid', JSON.stringify({
    rows: gridRows.value,
    cols: gridCols.value,
    order,
  }))
  formData.append('model_id', currentModel.value)
  formData.append('overlap', overlap.value)

  try {
    const { data } = await stitchAndDetect(formData)
    result.value = data
    stage.value = 'result'
    await nextTick()
    zoomFit()
    if (data.summary.total_instances === 0) {
      ElMessage.info('未检测到缺陷')
    }
  } catch (err) {
    const msg = err.response?.data?.detail || err.message || '拼接检测失败'
    ElMessage.error(msg)
  } finally {
    stitching.value = false
  }
}

// ---- 阶段2: 缩放平移 ----
const viewportRef = ref(null)
const overlayCanvas = ref(null)

const scale = ref(1)
const translateX = ref(0)
const translateY = ref(0)
let isPanning = false
let panStartX = 0
let panStartY = 0
let panStartTX = 0
let panStartTY = 0

const transformStyle = computed(() => ({
  transform: `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`,
  transformOrigin: '0 0',
}))

onMounted(async () => {
  try {
    const { data } = await getModels()
    models.value = data.models
  } catch {
    models.value = [{ id: 'segformer-b2', name: 'SegFormer-B2' }]
  }
})

function onImageLoad(e) {
  syncOverlayCanvas(e.target)
  drawOverlay()
}

function syncOverlayCanvas(imgEl) {
  const canvas = overlayCanvas.value
  if (!canvas || !imgEl || !result.value) return
  const { width, height } = result.value.image
  canvas.width = width
  canvas.height = height
  canvas.style.width = imgEl.offsetWidth + 'px'
  canvas.style.height = imgEl.offsetHeight + 'px'
}

function drawOverlay() {
  const canvas = overlayCanvas.value
  if (!canvas || !result.value) return
  const { width, height } = result.value.image

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, width, height)

  for (const inst of result.value.instances) {
    const key = `${inst.class_name}_${inst.instance_id}`
    const isHighlighted = key === highlightedId.value
    const color = result.value.classes[inst.class_name]?.color_rgb || [255, 255, 255]
    const colorStr = `rgb(${color[0]},${color[1]},${color[2]})`

    drawContour(ctx, inst, colorStr, isHighlighted)
    drawLabel(ctx, inst, colorStr, isHighlighted)
  }
}

function drawContour(ctx, inst, colorStr, isHighlighted) {
  const contour = inst.contour
  if (!contour || contour.length < 3) return

  const path = new Path2D()
  path.moveTo(contour[0][0], contour[0][1])
  for (let i = 1; i < contour.length; i++) {
    path.lineTo(contour[i][0], contour[i][1])
  }
  path.closePath()

  ctx.save()
  ctx.globalAlpha = isHighlighted ? 0.6 : 0.45
  ctx.fillStyle = colorStr
  ctx.fill(path)
  ctx.restore()

  ctx.save()
  ctx.globalAlpha = isHighlighted ? 1.0 : 0.85
  ctx.strokeStyle = colorStr
  ctx.lineWidth = isHighlighted ? 3 : 2
  ctx.stroke(path)
  ctx.restore()
}

function drawLabel(ctx, inst, colorStr, isHighlighted) {
  const cx = inst.centroid.x
  const cy = inst.centroid.y
  const text = `${inst.class_name_cn}#${inst.instance_id}`

  ctx.save()
  ctx.font = `${isHighlighted ? 'bold 18px' : '16px'} "Microsoft YaHei", sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.strokeStyle = '#000'
  ctx.lineWidth = 3
  ctx.globalAlpha = 0.8
  ctx.strokeText(text, cx, cy)
  ctx.fillStyle = isHighlighted ? '#fff' : colorStr
  ctx.globalAlpha = 1.0
  ctx.fillText(text, cx, cy)
  ctx.restore()
}

// 缩放
function onWheel(e) {
  const viewport = viewportRef.value
  if (!viewport) return

  const rect = viewport.getBoundingClientRect()
  const mouseX = e.clientX - rect.left
  const mouseY = e.clientY - rect.top

  const delta = e.deltaY > 0 ? 0.9 : 1.1
  const newScale = Math.max(0.05, Math.min(20, scale.value * delta))

  // 以鼠标位置为中心缩放
  translateX.value = mouseX - (mouseX - translateX.value) * (newScale / scale.value)
  translateY.value = mouseY - (mouseY - translateY.value) * (newScale / scale.value)
  scale.value = newScale
}

// 平移
function onPanStart(e) {
  if (e.button !== 0) return
  isPanning = true
  panStartX = e.clientX
  panStartY = e.clientY
  panStartTX = translateX.value
  panStartTY = translateY.value
}

function onPanMove(e) {
  if (!isPanning) return
  translateX.value = panStartTX + (e.clientX - panStartX)
  translateY.value = panStartTY + (e.clientY - panStartY)
}

function onPanEnd() {
  isPanning = false
}

function zoomFit() {
  const viewport = viewportRef.value
  if (!viewport || !result.value) return

  const { width, height } = result.value.image
  const vw = viewport.clientWidth
  const vh = viewport.clientHeight

  const s = Math.min(vw / width, vh / height) * 0.95
  scale.value = s
  translateX.value = (vw - width * s) / 2
  translateY.value = (vh - height * s) / 2
}

function zoomReset() {
  scale.value = 1
  translateX.value = 0
  translateY.value = 0
}

function backToPreview() {
  stage.value = 'preview'
  result.value = null
  highlightedId.value = null
}

watch(highlightedId, () => nextTick(drawOverlay))
watch(renderMode, () => nextTick(drawOverlay))
</script>
