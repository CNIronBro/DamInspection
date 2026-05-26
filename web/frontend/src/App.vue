<template>
  <el-container style="height: 100vh">
    <el-aside width="220px">
      <Sidebar :model-ready="modelReady" />
    </el-aside>
    <el-container>
      <el-main style="padding: 0; overflow: hidden">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import { getModels } from './api'

const modelReady = ref(false)

onMounted(async () => {
  try {
    const { data } = await getModels()
    modelReady.value = data.ready
  } catch {
    // ignore
  }
})
</script>
