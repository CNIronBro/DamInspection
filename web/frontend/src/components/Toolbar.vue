<template>
  <div class="toolbar">
    <div class="toolbar-left">
      <el-select
        :model-value="currentModel"
        @update:model-value="$emit('update:currentModel', $event)"
        placeholder="选择模型"
        style="width: 200px"
        size="default"
        :disabled="editMode"
      >
        <el-option
          v-for="m in models"
          :key="m.id"
          :label="m.name"
          :value="m.id"
          :disabled="!m.available"
        >
          <div>
            <div style="font-weight: 500">{{ m.name }}</div>
            <div style="font-size: 12px; color: #909399">{{ m.description }}</div>
          </div>
        </el-option>
      </el-select>

      <el-upload
        :show-file-list="false"
        accept="image/*"
        :before-upload="onBeforeUpload"
        :disabled="detecting || editMode"
      >
        <el-button type="primary" :icon="Upload" :loading="detecting">
          {{ detecting ? '分析中...' : '上传图片' }}
        </el-button>
      </el-upload>
    </div>

    <div class="toolbar-right">
      <el-button
        v-if="hasResult && !editMode"
        type="warning"
        :icon="Edit"
        @click="$emit('enter-edit')"
      >
        编辑标注
      </el-button>

      <template v-if="editMode">
        <el-button type="success" :icon="Check" @click="$emit('save')">
          保存
        </el-button>
        <el-button :icon="Close" @click="$emit('exit-edit')">
          取消
        </el-button>
      </template>

      <el-radio-group
        :model-value="renderMode"
        @update:model-value="$emit('update:renderMode', $event)"
        size="default"
      >
        <el-radio-button value="contour">标注模式</el-radio-button>
        <el-radio-button value="pixel">纯净模式</el-radio-button>
      </el-radio-group>
    </div>
  </div>
</template>

<script setup>
import { Upload, Edit, Check, Close } from '@element-plus/icons-vue'

defineProps({
  models: { type: Array, default: () => [] },
  currentModel: { type: String, default: 'segformer-b2' },
  renderMode: { type: String, default: 'contour' },
  detecting: { type: Boolean, default: false },
  hasResult: { type: Boolean, default: false },
  editMode: { type: Boolean, default: false },
})

const emit = defineEmits([
  'update:currentModel', 'update:renderMode', 'upload',
  'enter-edit', 'exit-edit', 'save',
])

function onBeforeUpload(file) {
  emit('upload', file)
  return false
}
</script>
