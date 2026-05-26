import { ref, reactive } from 'vue'
import { measureAnnotation } from '../api'

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

  function setTool(t) {
    tool.value = t
    drawingState.active = false
    drawingState.points = []
    drawingState.previewPoint = null
    selectedId.value = null
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

  function clear() {
    annotations.value = []
    selectedId.value = null
    drawingState.active = false
    drawingState.points = []
    drawingState.previewPoint = null
    nextId = 1
  }

  return {
    annotations,
    tool,
    className,
    selectedId,
    drawingState,
    setTool,
    addAnnotation,
    updateAnnotation,
    selectAnnotation,
    updateDrawing,
    clear,
  }
}
