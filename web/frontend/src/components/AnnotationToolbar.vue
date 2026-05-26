<template>
  <div class="annotation-toolbar">
    <div class="ann-group">
      <span class="ann-label">工具</span>
      <el-radio-group :model-value="tool" @update:model-value="$emit('update:tool', $event)" size="small">
        <el-radio-button value="pointer">
          <el-icon><Aim /></el-icon> 指针
        </el-radio-button>
        <el-radio-button value="polygon">
          <el-icon><Share /></el-icon> 多边形
        </el-radio-button>
        <el-radio-button value="rectangle">
          <el-icon><Grid /></el-icon> 矩形
        </el-radio-button>
      </el-radio-group>
    </div>

    <div class="ann-group">
      <span class="ann-label">类别</span>
      <el-select
        :model-value="className"
        @update:model-value="$emit('update:className', $event)"
        size="small"
        style="width: 120px"
      >
        <el-option value="crack" label="裂缝" />
        <el-option value="spalling" label="剥落" />
        <el-option value="efflorescence" label="泛碱" />
      </el-select>
    </div>

    <div class="ann-tip">
      <template v-if="tool === 'polygon'">点击添加顶点，双击或点击首点闭合</template>
      <template v-else-if="tool === 'rectangle'">按住拖拽绘制矩形</template>
      <template v-else>点击已有标注可选中编辑</template>
    </div>
  </div>
</template>

<script setup>
import { Aim, Share, Grid } from '@element-plus/icons-vue'

defineProps({
  tool: { type: String, default: 'pointer' },
  className: { type: String, default: 'crack' },
})

defineEmits(['update:tool', 'update:className'])
</script>
