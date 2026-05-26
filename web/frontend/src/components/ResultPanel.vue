<template>
  <div class="result-panel">
    <div class="panel-header">检测结果</div>
    <div class="panel-body">
      <!-- 汇总卡片 -->
      <div class="summary-cards">
        <div class="summary-card">
          <div class="count" style="color: #909399">{{ summary?.by_class?.crack || 0 }}</div>
          <div class="label">裂缝</div>
        </div>
        <div class="summary-card">
          <div class="count" style="color: #409eff">{{ summary?.by_class?.spalling || 0 }}</div>
          <div class="label">剥落</div>
        </div>
        <div class="summary-card">
          <div class="count" style="color: #e6a23c">{{ summary?.by_class?.efflorescence || 0 }}</div>
          <div class="label">泛碱</div>
        </div>
      </div>

      <!-- 空状态 -->
      <el-empty
        v-if="totalDisplayCount === 0"
        description="未检测到缺陷"
        :image-size="80"
      />

      <!-- 实例表格 -->
      <div v-else>
        <div style="margin-bottom: 10px; font-size: 13px; color: #606266">
          共 <b>{{ totalDisplayCount }}</b> 个缺陷实例
          <span v-if="editMode" style="color: #e6a23c; margin-left: 8px">（编辑模式）</span>
        </div>
        <div style="overflow-x: auto">
          <table class="instance-table">
            <thead>
              <tr>
                <th>来源</th>
                <th>类别</th>
                <th>#</th>
                <th>面积(mm²)</th>
                <th v-if="hasCrack">长度(mm)</th>
                <th v-if="hasCrack">宽度(mm)</th>
                <th v-if="hasSpallingOrEff">直径(mm)</th>
                <th v-if="editMode" style="width: 50px">操作</th>
              </tr>
            </thead>
            <tbody>
              <!-- 算法实例 -->
              <tr
                v-for="inst in instances"
                :key="getInstanceKey(inst)"
                :class="{
                  highlighted: getInstanceKey(inst) === highlightedId,
                  'is-deleted': deletedKeys.has(getInstanceKey(inst)),
                }"
                @mouseenter="$emit('hoverInstance', getInstanceKey(inst))"
                @mouseleave="$emit('hoverInstance', null)"
              >
                <td>
                  <span class="source-tag" :class="inst.source === 'manual' ? 'manual' : 'auto'">
                    {{ inst.source === 'manual' ? '人工' : '算法' }}
                  </span>
                </td>
                <td>
                  <span class="class-tag" :class="inst.class_name">
                    {{ inst.class_name_cn }}
                  </span>
                </td>
                <td>{{ inst.instance_id }}</td>
                <td>{{ inst.metrics?.area_mm2?.toFixed(1) || '-' }}</td>
                <td v-if="hasCrack">
                  {{ inst.class_name === 'crack' ? (inst.metrics?.length_mm?.toFixed(1) || '-') : '-' }}
                </td>
                <td v-if="hasCrack">
                  {{ inst.class_name === 'crack' ? (inst.metrics?.width_mean_mm?.toFixed(2) || '-') : '-' }}
                </td>
                <td v-if="hasSpallingOrEff">
                  {{ inst.class_name !== 'crack' ? (inst.metrics?.eq_diameter_mm?.toFixed(1) || '-') : '-' }}
                </td>
                <td v-if="editMode">
                  <el-button
                    type="danger"
                    link
                    size="small"
                    @click.stop="$emit('toggleDelete', getInstanceKey(inst))"
                  >
                    {{ deletedKeys.has(getInstanceKey(inst)) ? '恢复' : '删除' }}
                  </el-button>
                </td>
              </tr>
              <!-- 手动标注 -->
              <tr
                v-for="ann in manualAnnotations"
                :key="ann.id"
                :class="{ 'is-deleted': deletedManualIds.has(ann.id) }"
                class="manual-row"
              >
                <td>
                  <span class="source-tag manual">人工</span>
                </td>
                <td>
                  <span class="class-tag" :class="ann.className">
                    {{ classLabelCn(ann.className) }}
                  </span>
                </td>
                <td>-</td>
                <td>{{ ann.metrics?.area_mm2?.toFixed(1) || '-' }}</td>
                <td v-if="hasCrack">
                  {{ ann.className === 'crack' ? (ann.metrics?.length_mm?.toFixed(1) || '-') : '-' }}
                </td>
                <td v-if="hasCrack">
                  {{ ann.className === 'crack' ? (ann.metrics?.width_mean_mm?.toFixed(2) || '-') : '-' }}
                </td>
                <td v-if="hasSpallingOrEff">
                  {{ ann.className !== 'crack' ? (ann.metrics?.eq_diameter_mm?.toFixed(1) || '-') : '-' }}
                </td>
                <td v-if="editMode">
                  <el-button
                    type="danger"
                    link
                    size="small"
                    @click.stop="$emit('toggleDeleteManual', ann.id)"
                  >
                    {{ deletedManualIds.has(ann.id) ? '恢复' : '删除' }}
                  </el-button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  instances: { type: Array, default: () => [] },
  summary: { type: Object, default: () => ({}) },
  modelName: { type: String, default: '' },
  highlightedId: { type: String, default: null },
  editMode: { type: Boolean, default: false },
  manualAnnotations: { type: Array, default: () => [] },
  deletedKeys: { type: Set, default: () => new Set() },
  deletedManualIds: { type: Set, default: () => new Set() },
})

defineEmits(['hoverInstance', 'toggleDelete', 'toggleDeleteManual'])

const totalDisplayCount = computed(() =>
  props.instances.length + props.manualAnnotations.length
)

const hasCrack = computed(() =>
  props.instances.some(i => i.class_name === 'crack') ||
  props.manualAnnotations.some(a => a.className === 'crack')
)

const hasSpallingOrEff = computed(() =>
  props.instances.some(i => i.class_name !== 'crack') ||
  props.manualAnnotations.some(a => a.className !== 'crack')
)

function getInstanceKey(inst) {
  return `${inst.class_name}_${inst.instance_id}`
}

function classLabelCn(name) {
  const map = { crack: '裂缝', spalling: '剥落', efflorescence: '泛碱' }
  return map[name] || name
}
</script>
