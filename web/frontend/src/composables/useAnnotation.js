import { ref, reactive } from 'vue'
import { measureAnnotation } from '../api'
import { maskToContour, simplifyContour } from './useContourExtract'

let nextId = 1

export function useAnnotation() {
  const annotations = ref([])
  const tool = ref('pointer')
  const className = ref('crack')
  const selectedId = ref(null)
  const drawingState = reactive({
    active: false,
    points: [],
    previewPoint: null,
  })

  // 画笔状态
  const brushStrokes = ref([])   // 已完成的笔画列表，每项是 [[x,y], ...]
  const brushWidth = ref(15)
  const isBrushing = ref(false)  // 是否正在画一笔

  function setTool(t) {
    tool.value = t
    drawingState.active = false
    drawingState.points = []
    drawingState.previewPoint = null
    selectedId.value = null
    if (t !== 'brush') {
      brushStrokes.value = []
      isBrushing.value = false
    }
  }

  async function addAnnotation(ann, imageWidth, imageHeight, mmPerPx) {
    let metrics = {}
    try {
      const { data } = await measureAnnotation({
        points: ann.points,
        class_name: ann.className,
        image_width: imageWidth,
        image_height: imageHeight,
        mm_per_px: mmPerPx,
      })
      metrics = data.metrics
    } catch (err) {
      console.error('量化测量失败:', err)
    }

    annotations.value.push({
      id: `manual_${nextId++}`,
      className: ann.className,
      points: ann.points,
      metrics,
    })
    drawingState.active = false
    drawingState.points = []
    drawingState.previewPoint = null
  }

  async function updateAnnotation(id, updates, imageWidth, imageHeight, mmPerPx) {
    const idx = annotations.value.findIndex(a => a.id === id)
    if (idx < 0) return
    const merged = { ...annotations.value[idx], ...updates }

    if (updates.points) {
      try {
        const { data } = await measureAnnotation({
          points: merged.points,
          class_name: merged.className,
          image_width: imageWidth,
          image_height: imageHeight,
          mm_per_px: mmPerPx,
        })
        merged.metrics = data.metrics
      } catch (err) {
        console.error('量化测量失败:', err)
      }
    }

    annotations.value[idx] = merged
  }

  function selectAnnotation(id) {
    selectedId.value = id
  }

  function updateDrawing(data) {
    if (data.points !== undefined) drawingState.points = data.points
    if (data.previewPoint !== undefined) drawingState.previewPoint = data.previewPoint
    if (data.active !== undefined) drawingState.active = data.active
  }

  // ---- 画笔操作 ----

  function addStroke(stroke) {
    if (stroke.length >= 2) {
      brushStrokes.value = [...brushStrokes.value, stroke]
    }
    isBrushing.value = false
  }

  function undoStroke() {
    if (brushStrokes.value.length > 0) {
      brushStrokes.value = brushStrokes.value.slice(0, -1)
    }
  }

  function clearStrokes() {
    brushStrokes.value = []
    isBrushing.value = false
  }

  async function confirmBrush(imageWidth, imageHeight, mmPerPx) {
    if (brushStrokes.value.length === 0) return

    // 创建临时 canvas 合并所有笔画
    const tmpCanvas = document.createElement('canvas')
    tmpCanvas.width = imageWidth
    tmpCanvas.height = imageHeight
    const ctx = tmpCanvas.getContext('2d')

    ctx.fillStyle = '#000'
    ctx.strokeStyle = '#000'
    ctx.lineWidth = brushWidth.value
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    for (const stroke of brushStrokes.value) {
      if (stroke.length < 2) continue
      ctx.beginPath()
      ctx.moveTo(stroke[0][0], stroke[0][1])
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo(stroke[i][0], stroke[i][1])
      }
      ctx.stroke()
    }

    // 读取像素，生成二值掩码
    const imageData = ctx.getImageData(0, 0, imageWidth, imageHeight)
    const mask = new Uint8Array(imageWidth * imageHeight)
    for (let i = 0; i < mask.length; i++) {
      mask[i] = imageData.data[i * 4 + 3] > 0 ? 1 : 0
    }

    // 提取轮廓 + 简化
    const rawContour = maskToContour(mask, imageWidth, imageHeight)
    if (rawContour.length < 3) {
      clearStrokes()
      return
    }
    const contour = simplifyContour(rawContour, 2)

    // 清除笔画状态
    clearStrokes()

    // 提交为 annotation（复用 addAnnotation 流程）
    await addAnnotation({ className: className.value, points: contour }, imageWidth, imageHeight, mmPerPx)
  }

  function clear() {
    annotations.value = []
    selectedId.value = null
    drawingState.active = false
    drawingState.points = []
    drawingState.previewPoint = null
    brushStrokes.value = []
    isBrushing.value = false
    nextId = 1
  }

  return {
    annotations,
    tool,
    className,
    selectedId,
    drawingState,
    brushStrokes,
    brushWidth,
    isBrushing,
    setTool,
    addAnnotation,
    updateAnnotation,
    selectAnnotation,
    updateDrawing,
    addStroke,
    undoStroke,
    clearStrokes,
    confirmBrush,
    clear,
  }
}
