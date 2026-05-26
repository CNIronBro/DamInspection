<template>
  <div class="detection-view">
    <Toolbar
      v-model:current-model="currentModel"
      v-model:render-mode="renderMode"
      :models="models"
      :detecting="detecting"
      :has-result="!!result"
      :edit-mode="editMode"
      @upload="handleUpload"
      @enter-edit="enterEditMode"
      @exit-edit="exitEditMode"
      @save="handleSave"
    />

    <AnnotationToolbar
      v-if="editMode"
      :tool="annTool"
      :class-name="annClassName"
      @update:tool="annSetTool($event)"
      @update:class-name="annClassName = $event"
    />

    <div class="main-content">
      <div class="canvas-area">
        <CanvasOverlay
          v-if="result"
          :image-data="result.image"
          :instances="displayInstances"
          :classes="result.classes"
          :render-mode="renderMode"
          :highlighted-id="highlightedId"
          :edit-mode="editMode"
          :manual-annotations="displayManualAnnotations"
          :active-tool="annTool"
          :active-class-name="annClassName"
          :drawing-state="annDrawingState"
          :selected-annotation-id="annSelectedId"
          @hover-instance="highlightedId = $event"
          @click-instance="handleClickInstance"
          @annotation-add="annAddAnnotation"
          @annotation-update="annUpdateAnnotation"
          @annotation-select="annSelectAnnotation"
          @drawing-update="annUpdateDrawing"
        />
        <div v-else class="empty-state">
          <el-icon :size="48" color="#c0c4cc"><Upload /></el-icon>
          <span>上传图片开始缺陷检测</span>
        </div>
        <div v-if="detecting" class="detecting-overlay">
          <div class="spinner"></div>
          <div class="text">正在分析...</div>
        </div>
      </div>
      <ResultPanel
        v-if="result"
        :instances="result.instances"
        :summary="result.summary"
        :model-name="currentModelName"
        :highlighted-id="highlightedId"
        :edit-mode="editMode"
        :manual-annotations="annAnnotations"
        :deleted-keys="deletedInstanceKeys"
        :deleted-manual-ids="deletedManualIds"
        @hover-instance="highlightedId = $event"
        @toggle-delete="handleToggleDelete"
        @toggle-delete-manual="handleToggleDeleteManual"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Upload } from '@element-plus/icons-vue'
import Toolbar from '../components/Toolbar.vue'
import AnnotationToolbar from '../components/AnnotationToolbar.vue'
import CanvasOverlay from '../components/CanvasOverlay.vue'
import ResultPanel from '../components/ResultPanel.vue'
import { getModels, detect as detectApi, saveRecord, getRecord } from '../api'
import { useAnnotation } from '../composables/useAnnotation'

const route = useRoute()

const models = ref([])
const currentModel = ref('segformer-b2')
const renderMode = ref('contour')
const detecting = ref(false)
const result = ref(null)
const highlightedId = ref(null)
const modelReady = ref(false)
const editMode = ref(false)
const currentRecordId = ref(0)
const deletedInstanceKeys = ref(new Set())
const deletedManualIds = ref(new Set())

const {
  annotations: annAnnotations,
  tool: annTool,
  className: annClassName,
  selectedId: annSelectedId,
  drawingState: annDrawingState,
  setTool: annSetTool,
  addAnnotation: _annAddAnnotation,
  updateAnnotation: _annUpdateAnnotation,
  selectAnnotation: annSelectAnnotation,
  updateDrawing: annUpdateDrawing,
  clear: annClear,
} = useAnnotation()

// 包装函数，传入图片尺寸和比例尺供后端量化计算
function annAddAnnotation(ann) {
  if (!result.value) return
  const { width, height } = result.value.image
  _annAddAnnotation(ann, width, height, 1.0)
}

function annUpdateAnnotation(id, updates) {
  if (!result.value) return
  const { width, height } = result.value.image
  _annUpdateAnnotation(id, updates, width, height, 1.0)
}

const currentModelName = computed(() => {
  const m = models.value.find(m => m.id === currentModel.value)
  return m ? m.name : currentModel.value
})

// 传给 CanvasOverlay 的实例列表（过滤掉已删除的）
const displayInstances = computed(() => {
  if (!result.value) return []
  if (!editMode.value) return result.value.instances
  return result.value.instances.filter(
    inst => !deletedInstanceKeys.value.has(`${inst.class_name}_${inst.instance_id}`)
  )
})

// 传给 CanvasOverlay 的手动标注（过滤掉已删除的）
const displayManualAnnotations = computed(() => {
  if (!editMode.value) return annAnnotations.value
  return annAnnotations.value.filter(ann => !deletedManualIds.value.has(ann.id))
})

onMounted(async () => {
  try {
    const { data } = await getModels()
    models.value = data.models
    modelReady.value = data.ready
  } catch {
    models.value = [{ id: 'segformer-b2', name: 'SegFormer-B2', available: true }]
  }

  const recordId = route.query.recordId
  if (recordId) {
    await loadRecord(Number(recordId))
  }
})

watch(() => route.query.recordId, (id) => {
  if (id) loadRecord(Number(id))
})

async function loadRecord(recordId) {
  try {
    const { data: record } = await getRecord(recordId)
    currentRecordId.value = recordId
    const instances = (record.instances || []).map(inst => ({
      class_name: inst.class_name,
      class_name_cn: inst.class_name_cn,
      instance_id: inst.instance_id,
      source: inst.source || 'auto',
      bbox: inst.bbox || inst.bbox_json,
      centroid: inst.centroid || inst.centroid_json,
      contour: inst.contour || inst.contour_json,
      metrics: {
        area_mm2: inst.area_mm2,
        length_mm: inst.length_mm,
        width_mean_mm: inst.width_mean_mm,
        eq_diameter_mm: inst.eq_diameter_mm,
      },
      rle: null,
    }))

    const classes = {
      crack: { color_rgb: [144, 147, 153], label_cn: '裂缝' },
      spalling: { color_rgb: [64, 158, 255], label_cn: '剥落' },
      efflorescence: { color_rgb: [230, 162, 60], label_cn: '泛碱' },
    }

    const byClass = record.by_class || {}
    result.value = {
      image: {
        width: record.image_width,
        height: record.image_height,
        path: record.image_path,
      },
      instances,
      classes,
      summary: {
        total_instances: record.total_instances,
        by_class: byClass,
      },
    }
    currentModel.value = record.model_id || 'segformer-b2'
  } catch (err) {
    ElMessage.error('加载记录失败')
  }
}

async function handleUpload(file) {
  detecting.value = true
  result.value = null
  highlightedId.value = null
  editMode.value = false
  currentRecordId.value = 0
  deletedInstanceKeys.value = new Set()
  deletedManualIds.value = new Set()
  annClear()

  const formData = new FormData()
  formData.append('file', file)
  formData.append('model_id', currentModel.value)

  try {
    const { data } = await detectApi(formData)
    result.value = data
    if (data.summary.total_instances === 0) {
      ElMessage.info('未检测到缺陷')
    }
  } catch (err) {
    const msg = err.response?.data?.detail || err.message || '检测失败'
    ElMessage.error(msg)
  } finally {
    detecting.value = false
  }
}

function enterEditMode() {
  editMode.value = true
  deletedInstanceKeys.value = new Set()
  deletedManualIds.value = new Set()
  annClear()
}

function exitEditMode() {
  editMode.value = false
  deletedInstanceKeys.value = new Set()
  deletedManualIds.value = new Set()
  annClear()
}

function handleClickInstance(key) {
  if (editMode.value && key) {
    handleToggleDelete(key)
  } else {
    highlightedId.value = key
  }
}

function handleToggleDelete(key) {
  const s = new Set(deletedInstanceKeys.value)
  if (s.has(key)) {
    s.delete(key)
  } else {
    s.add(key)
  }
  deletedInstanceKeys.value = s
}

function handleToggleDeleteManual(annId) {
  const s = new Set(deletedManualIds.value)
  if (s.has(annId)) {
    s.delete(annId)
  } else {
    s.add(annId)
  }
  deletedManualIds.value = s
}

async function handleSave() {
  if (!result.value) return

  // 过滤掉被删除的算法实例
  const autoInstances = result.value.instances.filter(
    inst => !deletedInstanceKeys.value.has(`${inst.class_name}_${inst.instance_id}`)
  )

  // 手动标注转为标准实例格式（过滤掉已删除的）
  const manualInstances = annAnnotations.value
    .filter(ann => !deletedManualIds.value.has(ann.id))
    .map((ann, idx) => {
    const pts = ann.points
    const xs = pts.map(p => p[0])
    const ys = pts.map(p => p[1])
    const xmin = Math.min(...xs)
    const ymin = Math.min(...ys)
    const xmax = Math.max(...xs)
    const ymax = Math.max(...ys)
    const cx = Math.round(xs.reduce((a, b) => a + b, 0) / xs.length)
    const cy = Math.round(ys.reduce((a, b) => a + b, 0) / ys.length)

    const m = ann.metrics || {}
    const classInfo = result.value.classes[ann.className] || {}
    return {
      class_name: ann.className,
      class_name_cn: classInfo.label_cn || ann.className,
      instance_id: 1000 + idx,
      source: 'manual',
      area_mm2: m.area_mm2 || 0,
      bbox: { xmin, ymin, xmax, ymax },
      centroid: { x: cx, y: cy },
      contour: pts.map(p => [p[0], p[1]]),
      metrics: {
        area_mm2: m.area_mm2 || 0,
        length_mm: m.length_mm || 0,
        width_mean_mm: m.width_mean_mm || 0,
        eq_diameter_mm: m.eq_diameter_mm || 0,
      },
    }
  })

  const allInstances = [...autoInstances, ...manualInstances]

  const byClass = {}
  for (const inst of allInstances) {
    byClass[inst.class_name] = (byClass[inst.class_name] || 0) + 1
  }

  const payload = {
    record_id: currentRecordId.value,
    image_name: 'uploaded_image',
    image_width: result.value.image.width,
    image_height: result.value.image.height,
    image_path: result.value.image.path || '',
    model_id: currentModel.value,
    mm_per_px: 1.0,
    total_instances: allInstances.length,
    by_class: byClass,
    instances: allInstances.map(inst => ({
      class_name: inst.class_name,
      class_name_cn: inst.class_name_cn,
      instance_id: inst.instance_id,
      source: inst.source || 'auto',
      area_mm2: inst.metrics?.area_mm2 || inst.area_mm2 || 0,
      length_mm: inst.metrics?.length_mm || inst.length_mm || 0,
      width_mean_mm: inst.metrics?.width_mean_mm || 0,
      width_max_mm: inst.metrics?.width_max_mm || 0,
      width_p95_mm: inst.metrics?.width_p95_mm || 0,
      eq_diameter_mm: inst.metrics?.eq_diameter_mm || 0,
      major_axis_mm: inst.metrics?.major_axis_mm || 0,
      minor_axis_mm: inst.metrics?.minor_axis_mm || 0,
      bbox: inst.bbox,
      contour: inst.contour,
      centroid: inst.centroid,
    })),
  }

  try {
    await saveRecord(payload)
    ElMessage.success('保存成功')
    // 保存后合并，保留原始 source
    const merged = allInstances.map(inst => ({ ...inst }))
    result.value = { ...result.value, instances: merged }
    editMode.value = false
    deletedInstanceKeys.value = new Set()
    deletedManualIds.value = new Set()
    annClear()
  } catch (err) {
    ElMessage.error('保存失败: ' + (err.message || '未知错误'))
  }
}
</script>
