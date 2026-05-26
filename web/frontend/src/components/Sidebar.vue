<template>
  <div class="sidebar">
    <div class="sidebar-brand">
      <h2>大坝缺陷检测</h2>
      <div class="subtitle">Dam Inspection AI</div>
    </div>

    <nav class="sidebar-nav">
      <router-link to="/" class="nav-item" :class="{ active: $route.path === '/' }">
        <el-icon class="icon"><Monitor /></el-icon>
        <span>缺陷检测</span>
      </router-link>
      <router-link to="/stats" class="nav-item" :class="{ active: $route.path === '/stats' }">
        <el-icon class="icon"><Clock /></el-icon>
        <span>历史记录</span>
      </router-link>
      <el-tooltip content="即将开放" placement="right" :show-after="300">
        <div class="nav-item disabled">
          <el-icon class="icon"><Document /></el-icon>
          <span>报告导出</span>
        </div>
      </el-tooltip>
      <el-tooltip content="即将开放" placement="right" :show-after="300">
        <div class="nav-item disabled">
          <el-icon class="icon"><Setting /></el-icon>
          <span>模型管理</span>
        </div>
      </el-tooltip>
    </nav>

    <div class="sidebar-footer">
      <span class="status-dot" :class="{ loading: !modelReady }"></span>
      <span>{{ modelReady ? '模型已就绪' : '模型加载中...' }}</span>
    </div>
  </div>
</template>

<script setup>
import { Monitor, Clock, Document, Setting } from '@element-plus/icons-vue'

defineProps({
  modelReady: { type: Boolean, default: false },
})
</script>
