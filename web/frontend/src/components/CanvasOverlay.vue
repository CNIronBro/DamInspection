<template>
  <div class="canvas-wrapper" ref="wrapperRef">
    <img
      v-if="imageSrc"
      :src="imageSrc"
      class="base-image"
      @load="onBaseImageLoad"
    />
    <canvas
      ref="overlayCanvas"
      class="canvas-overlay"
      :style="{ cursor: editMode ? activeCursor : 'crosshair' }"
      @mousemove="onMouseMove"
      @mouseleave="onMouseLeave"
      @click="onClick"
      @dblclick="onDblClick"
      @mousedown="onMouseDown"
      @mouseup="onMouseUp"
    ></canvas>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, computed } from 'vue'

const props = defineProps({
  imageData: { type: Object, required: true },
  instances: { type: Array, default: () => [] },
  classes: { type: Object, default: () => ({}) },
  renderMode: { type: String, default: 'contour' },
  highlightedId: { type: String, default: null },
  editMode: { type: Boolean, default: false },
  manualAnnotations: { type: Array, default: () => [] },
  activeTool: { type: String, default: 'pointer' },
  activeClassName: { type: String, default: 'crack' },
  drawingState: { type: Object, default: () => ({ active: false, points: [], previewPoint: null }) },
  selectedAnnotationId: { type: String, default: null },
  brushStrokes: { type: Array, default: () => [] },
  brushWidth: { type: Number, default: 15 },
})

const emit = defineEmits([
  'hover-instance', 'click-instance',
  'annotation-add', 'annotation-update', 'annotation-select',
  'drawing-update',
  'brush-add-stroke',
])

const wrapperRef = ref(null)
const overlayCanvas = ref(null)

const MANUAL_COLOR = '#FF8C00'

// 画笔状态（本地）
let brushActive = false
let brushPoints = []
let brushMousePos = null  // 用于光标预览

const imageSrc = computed(() => {
  if (!props.imageData) return ''
  if (props.imageData.path) return props.imageData.path
  if (props.imageData.base64) {
    const b64 = props.imageData.base64
    const mime = b64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg'
    return `data:${mime};base64,${b64}`
  }
  return ''
})

const activeCursor = computed(() => {
  if (props.activeTool === 'polygon') return 'crosshair'
  if (props.activeTool === 'rectangle') return 'crosshair'
  if (props.activeTool === 'brush') return 'none'
  return 'default'
})

let isDragging = false
let dragVertexIdx = -1
let dragAnnotationId = null
let rectStart = null

function getInstanceKey(inst) {
  return `${inst.class_name}_${inst.instance_id}`
}

function getInstanceByKey(key) {
  return props.instances.find(i => getInstanceKey(i) === key)
}

// 坐标转换：DOM → canvas 像素
function toCanvasCoords(mx, my) {
  const canvas = overlayCanvas.value
  if (!canvas) return { x: 0, y: 0 }
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height
  return { x: mx * scaleX, y: my * scaleY }
}

// 图片加载完成后设置 canvas 尺寸并绘制 overlay
function onBaseImageLoad(e) {
  syncCanvasSize(e.target)
  drawOverlay()
}

// 同步 overlay canvas 的内部尺寸和 CSS 尺寸到 img 的实际显示尺寸
function syncCanvasSize(imgEl) {
  const canvas = overlayCanvas.value
  if (!canvas || !imgEl) return
  const { width, height } = props.imageData
  canvas.width = width
  canvas.height = height
  canvas.style.width = imgEl.offsetWidth + 'px'
  canvas.style.height = imgEl.offsetHeight + 'px'
}

// 绘制 overlay
function drawOverlay() {
  const canvas = overlayCanvas.value
  if (!canvas || !props.imageData) return
  const { width, height } = props.imageData

  // 同步 canvas 尺寸到 img 显示尺寸
  const wrapper = wrapperRef.value
  const imgEl = wrapper ? wrapper.querySelector('img') : null
  if (imgEl && imgEl.offsetWidth > 0) {
    canvas.width = width
    canvas.height = height
    canvas.style.width = imgEl.offsetWidth + 'px'
    canvas.style.height = imgEl.offsetHeight + 'px'
  }

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, width, height)

  if (!props.instances || props.instances.length === 0) {
    if (props.editMode) drawManualAnnotations(ctx)
    if (props.editMode && props.activeTool === 'brush') drawBrushOverlay(ctx)
    return
  }

  // 像素模式
  if (props.renderMode === 'pixel') {
    const imgData = ctx.createImageData(width, height)
    for (const inst of props.instances) {
      const key = getInstanceKey(inst)
      const isHighlighted = key === props.highlightedId
      const color = props.classes[inst.class_name]?.color_rgb || [255, 255, 255]
      writePixelMask(imgData, inst, color, isHighlighted)
    }
    ctx.putImageData(imgData, 0, 0)
  }

  // 轮廓模式
  for (const inst of props.instances) {
    const key = getInstanceKey(inst)
    const isHighlighted = key === props.highlightedId
    const color = props.classes[inst.class_name]?.color_rgb || [255, 255, 255]
    const colorStr = `rgb(${color[0]},${color[1]},${color[2]})`

    if (props.renderMode === 'contour') {
      drawContour(ctx, inst, colorStr, isHighlighted)
    }

    if (props.renderMode === 'contour') {
      drawLabel(ctx, inst, colorStr, isHighlighted, inst.class_name_cn, inst.instance_id)
    }
  }

  // 高亮 bbox
  if (props.highlightedId) {
    const inst = getInstanceByKey(props.highlightedId)
    if (inst) {
      drawBBox(ctx, inst, `rgb(${props.classes[inst.class_name]?.color_rgb?.join(',') || '255,255,255'})`)
    }
  }

  // 编辑模式：手动标注
  if (props.editMode) {
    drawManualAnnotations(ctx)
  }

  // 画笔覆盖层
  if (props.editMode && props.activeTool === 'brush') {
    drawBrushOverlay(ctx)
  }
}

// ---- 算法实例绘制 ----

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

function writePixelMask(imageData, inst, color, isHighlighted) {
  const rle = inst.rle
  if (!rle) return

  const data = imageData.data
  const alpha = isHighlighted ? 180 : 130
  const w = imageData.width

  let idx = 0
  let val = rle.start_with
  for (const count of rle.counts) {
    if (val === 1) {
      for (let i = 0; i < count; i++) {
        const offset = idx * 4
        if (data[offset + 3] < alpha) {
          data[offset] = color[0]
          data[offset + 1] = color[1]
          data[offset + 2] = color[2]
          data[offset + 3] = alpha
        }
        idx++
      }
    } else {
      idx += count
    }
    val = 1 - val
  }
}

function drawLabel(ctx, inst, colorStr, isHighlighted, nameCn, instId) {
  const cx = inst.centroid.x
  const cy = inst.centroid.y
  const text = `${nameCn}#${instId}`

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

function drawBBox(ctx, inst, colorStr) {
  const { xmin, ymin, xmax, ymax } = inst.bbox
  ctx.save()
  ctx.strokeStyle = colorStr
  ctx.lineWidth = 2
  ctx.setLineDash([6, 4])
  ctx.globalAlpha = 0.9
  ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin)
  ctx.restore()
}

// ---- 手动标注绘制 ----

function drawManualAnnotations(ctx) {
  for (const ann of props.manualAnnotations) {
    const isSelected = ann.id === props.selectedAnnotationId
    drawManualShape(ctx, ann.points, isSelected, false)
  }

  if (props.drawingState.active && props.drawingState.points.length > 0) {
    const pts = [...props.drawingState.points]
    if (props.drawingState.previewPoint) {
      pts.push(props.drawingState.previewPoint)
    }
    drawManualShape(ctx, pts, false, true)
  }
}

function drawManualShape(ctx, points, isSelected, isDrawing) {
  if (!points || points.length < 2) return

  ctx.save()

  ctx.globalAlpha = isSelected ? 0.55 : 0.4
  ctx.fillStyle = MANUAL_COLOR
  ctx.beginPath()
  ctx.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1])
  }
  if (!isDrawing) ctx.closePath()
  ctx.fill()

  ctx.globalAlpha = 1.0
  ctx.strokeStyle = MANUAL_COLOR
  ctx.lineWidth = isSelected ? 3 : 2
  ctx.setLineDash(isDrawing ? [6, 4] : [])
  ctx.beginPath()
  ctx.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1])
  }
  if (!isDrawing) ctx.closePath()
  ctx.stroke()
  ctx.setLineDash([])

  if (isSelected || isDrawing) {
    for (const [px, py] of points) {
      ctx.fillStyle = '#fff'
      ctx.strokeStyle = MANUAL_COLOR
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(px, py, 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }
  }

  if (!isDrawing && points.length >= 3) {
    const xs = points.map(p => p[0])
    const ys = points.map(p => p[1])
    const cx = xs.reduce((a, b) => a + b, 0) / xs.length
    const cy = ys.reduce((a, b) => a + b, 0) / ys.length

    const classInfo = props.classes[props.activeClassName] || {}
    const label = classInfo.label_cn || props.activeClassName

    ctx.font = 'bold 16px "Microsoft YaHei", sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 3
    ctx.globalAlpha = 0.8
    ctx.strokeText(`[人工]${label}`, cx, cy)
    ctx.fillStyle = MANUAL_COLOR
    ctx.globalAlpha = 1.0
    ctx.fillText(`[人工]${label}`, cx, cy)
  }

  ctx.restore()
}

// ---- 画笔绘制 ----

function drawBrushStroke(ctx, points, width) {
  if (!points || points.length < 2) return
  ctx.save()
  ctx.globalAlpha = 0.6
  ctx.strokeStyle = MANUAL_COLOR
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  ctx.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1])
  }
  ctx.stroke()
  ctx.restore()
}

function drawBrushOverlay(ctx) {
  // 已完成的笔画
  for (const stroke of props.brushStrokes) {
    drawBrushStroke(ctx, stroke, props.brushWidth)
  }

  // 正在画的笔画
  if (brushActive && brushPoints.length >= 2) {
    drawBrushStroke(ctx, brushPoints, props.brushWidth)
  }

  // 光标圆圈预览
  if (brushMousePos) {
    ctx.save()
    ctx.globalAlpha = 0.7
    ctx.strokeStyle = MANUAL_COLOR
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(brushMousePos[0], brushMousePos[1], props.brushWidth / 2, 0, Math.PI * 2)
    ctx.stroke()
    ctx.restore()
  }
}

// ---- 鼠标交互 ----

function hitTest(mx, my) {
  const { x: cx, y: cy } = toCanvasCoords(mx, my)

  for (let i = props.instances.length - 1; i >= 0; i--) {
    const inst = props.instances[i]
    const { xmin, ymin, xmax, ymax } = inst.bbox
    if (cx >= xmin && cx <= xmax && cy >= ymin && cy <= ymax) {
      return getInstanceKey(inst)
    }
  }
  return null
}

function hitTestManual(mx, my) {
  const { x: cx, y: cy } = toCanvasCoords(mx, my)

  for (let i = props.manualAnnotations.length - 1; i >= 0; i--) {
    const ann = props.manualAnnotations[i]
    const pts = ann.points
    if (!pts || pts.length < 3) continue

    const xs = pts.map(p => p[0])
    const ys = pts.map(p => p[1])
    const xmin = Math.min(...xs), xmax = Math.max(...xs)
    const ymin = Math.min(...ys), ymax = Math.max(...ys)

    if (cx >= xmin && cx <= xmax && cy >= ymin && cy <= ymax) {
      return ann.id
    }
  }
  return null
}

function hitTestVertex(mx, my) {
  const { x: cx, y: cy } = toCanvasCoords(mx, my)
  const threshold = 8

  for (const ann of props.manualAnnotations) {
    for (let i = 0; i < ann.points.length; i++) {
      const [px, py] = ann.points[i]
      const dist = Math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
      if (dist < threshold) {
        return { annotationId: ann.id, vertexIndex: i }
      }
    }
  }
  return null
}

function onMouseMove(e) {
  // 画笔模式
  if (props.editMode && props.activeTool === 'brush') {
    const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
    brushMousePos = [Math.round(x), Math.round(y)]

    if (brushActive) {
      brushPoints.push([Math.round(x), Math.round(y)])
    }

    drawOverlay()
    return
  }

  if (isDragging && dragAnnotationId !== null && dragVertexIdx >= 0) {
    const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
    const ann = props.manualAnnotations.find(a => a.id === dragAnnotationId)
    if (ann) {
      const newPoints = [...ann.points]
      newPoints[dragVertexIdx] = [Math.round(x), Math.round(y)]
      emit('annotation-update', dragAnnotationId, { points: newPoints })
    }
    return
  }

  if (props.editMode) {
    if (props.drawingState.active) {
      const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
      emit('drawing-update', { previewPoint: [Math.round(x), Math.round(y)] })
    }

    if (props.activeTool === 'rectangle' && rectStart) {
      const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
      const pts = [
        rectStart,
        [Math.round(x), rectStart[1]],
        [Math.round(x), Math.round(y)],
        [rectStart[0], Math.round(y)],
      ]
      emit('drawing-update', { points: pts, active: true, previewPoint: null })
    }

    return
  }

  const key = hitTest(e.offsetX, e.offsetY)
  emit('hover-instance', key)
}

function onMouseLeave() {
  if (props.editMode && props.activeTool === 'brush') {
    brushMousePos = null
    drawOverlay()
    return
  }

  if (!props.editMode) {
    emit('hover-instance', null)
  }
  if (props.drawingState.active) {
    emit('drawing-update', { previewPoint: null })
  }
}

function onClick(e) {
  if (!props.editMode) {
    const key = hitTest(e.offsetX, e.offsetY)
    if (key) emit('click-instance', key)
    return
  }

  // 画笔模式不处理 click
  if (props.activeTool === 'brush') return

  const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)

  if (props.activeTool === 'pointer') {
    const vtx = hitTestVertex(e.offsetX, e.offsetY)
    if (vtx) return

    const annId = hitTestManual(e.offsetX, e.offsetY)
    emit('annotation-select', annId)
    return
  }

  if (props.activeTool === 'polygon') {
    const point = [Math.round(x), Math.round(y)]

    if (!props.drawingState.active) {
      emit('drawing-update', { active: true, points: [point], previewPoint: null })
      return
    }

    const first = props.drawingState.points[0]
    const dist = Math.sqrt((x - first[0]) ** 2 + (y - first[1]) ** 2)
    if (dist < 10 && props.drawingState.points.length >= 3) {
      emit('annotation-add', {
        className: props.activeClassName,
        points: props.drawingState.points,
      })
      return
    }

    emit('drawing-update', {
      points: [...props.drawingState.points, point],
    })
    return
  }
}

function onDblClick(e) {
  if (!props.editMode) return

  if (props.activeTool === 'polygon' && props.drawingState.active) {
    if (props.drawingState.points.length >= 3) {
      emit('annotation-add', {
        className: props.activeClassName,
        points: props.drawingState.points,
      })
    }
    return
  }

  if (props.activeTool === 'pointer' && props.selectedAnnotationId) {
    const vtx = hitTestVertex(e.offsetX, e.offsetY)
    if (vtx) {
      const ann = props.manualAnnotations.find(a => a.id === vtx.annotationId)
      if (ann && ann.points.length > 3) {
        const newPoints = ann.points.filter((_, i) => i !== vtx.vertexIndex)
        emit('annotation-update', vtx.annotationId, { points: newPoints })
      }
    }
  }
}

function onMouseDown(e) {
  if (!props.editMode) return

  // 画笔模式：开始画一笔
  if (props.activeTool === 'brush') {
    const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
    brushActive = true
    brushPoints = [[Math.round(x), Math.round(y)]]
    return
  }

  if (props.activeTool === 'pointer') {
    const vtx = hitTestVertex(e.offsetX, e.offsetY)
    if (vtx) {
      isDragging = true
      dragAnnotationId = vtx.annotationId
      dragVertexIdx = vtx.vertexIndex
      return
    }
  }

  if (props.activeTool === 'rectangle') {
    const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
    rectStart = [Math.round(x), Math.round(y)]
  }
}

function onMouseUp(e) {
  // 画笔模式：结束一笔
  if (props.editMode && props.activeTool === 'brush' && brushActive) {
    brushActive = false
    if (brushPoints.length >= 2) {
      emit('brush-add-stroke', [...brushPoints])
    }
    brushPoints = []
    drawOverlay()
    return
  }

  if (isDragging) {
    isDragging = false
    dragAnnotationId = null
    dragVertexIdx = -1
    return
  }

  if (props.activeTool === 'rectangle' && rectStart) {
    const { x, y } = toCanvasCoords(e.offsetX, e.offsetY)
    const end = [Math.round(x), Math.round(y)]

    const pts = [
      rectStart,
      [end[0], rectStart[1]],
      end,
      [rectStart[0], end[1]],
    ]

    const w = Math.abs(end[0] - rectStart[0])
    const h = Math.abs(end[1] - rectStart[1])
    if (w > 5 && h > 5) {
      emit('annotation-add', {
        className: props.activeClassName,
        points: pts,
      })
    }

    rectStart = null
    emit('drawing-update', { active: false, points: [], previewPoint: null })
  }
}

// 监听变化重绘
watch(
  () => [props.instances, props.renderMode, props.highlightedId, props.editMode, props.manualAnnotations, props.drawingState, props.selectedAnnotationId, props.brushStrokes, props.brushWidth],
  () => nextTick(drawOverlay),
  { deep: true }
)
</script>
