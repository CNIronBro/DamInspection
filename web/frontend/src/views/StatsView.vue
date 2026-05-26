<template>
  <div class="stats-view">
    <div class="stats-header">
      <h2 style="margin: 0; font-size: 18px">检测历史与统计</h2>
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        range-separator="至"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        size="default"
        value-format="YYYY-MM-DD"
        @change="fetchStats"
      />
      <el-select v-model="filterClass" clearable placeholder="全部类型" size="default" @change="fetchStats" style="width: 130px">
        <el-option value="crack" label="裂缝" />
        <el-option value="spalling" label="剥落" />
        <el-option value="efflorescence" label="泛碱" />
      </el-select>
    </div>

    <!-- 汇总卡片 -->
    <div class="stats-cards">
      <div class="stat-card">
        <div class="value">{{ stats.total_records || 0 }}</div>
        <div class="label">检测记录数</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color: #909399">{{ classCount('crack') }}</div>
        <div class="label">裂缝</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color: #409eff">{{ classCount('spalling') }}</div>
        <div class="label">剥落</div>
      </div>
      <div class="stat-card">
        <div class="value" style="color: #e6a23c">{{ classCount('efflorescence') }}</div>
        <div class="label">泛碱</div>
      </div>
    </div>

    <!-- 图表 -->
    <div class="stats-charts">
      <div class="chart-box">
        <div class="chart-title">缺陷类型分布</div>
        <div ref="pieChartRef" style="height: 300px"></div>
      </div>
      <div class="chart-box">
        <div class="chart-title">缺陷类型计数</div>
        <div ref="barChartRef" style="height: 300px"></div>
      </div>
      <div class="chart-box" style="grid-column: 1 / -1">
        <div class="chart-title">缺陷趋势（按日）</div>
        <div ref="lineChartRef" style="height: 300px"></div>
      </div>
    </div>

    <!-- 历史记录表格 -->
    <div class="records-table-wrap">
      <div style="margin-bottom: 12px; font-size: 14px; font-weight: 600; color: #303133">
        历史记录
      </div>
      <el-table :data="records" stripe style="width: 100%" v-loading="loading">
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="image_name" label="图片名称" />
        <el-table-column prop="model_id" label="模型" width="140" />
        <el-table-column prop="total_instances" label="缺陷数" width="90" align="center" />
        <el-table-column label="裂缝" width="70" align="center">
          <template #default="{ row }">{{ row.by_class?.crack || 0 }}</template>
        </el-table-column>
        <el-table-column label="剥落" width="70" align="center">
          <template #default="{ row }">{{ row.by_class?.spalling || 0 }}</template>
        </el-table-column>
        <el-table-column label="泛碱" width="70" align="center">
          <template #default="{ row }">{{ row.by_class?.efflorescence || 0 }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="时间" width="180">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="100" align="center">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="viewRecord(row.id)">查看</el-button>
            <el-popconfirm title="确定删除?" @confirm="handleDelete(row.id)">
              <template #reference>
                <el-button type="danger" link size="small">删除</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
      <div style="margin-top: 16px; display: flex; justify-content: center">
        <el-pagination
          v-model:current-page="page"
          :page-size="pageSize"
          :total="total"
          layout="prev, pager, next"
          @current-change="fetchRecords"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { listRecords, deleteRecord, getStatsSummary } from '../api'

const router = useRouter()

const records = ref([])
const page = ref(1)
const pageSize = 20
const total = ref(0)
const loading = ref(false)
const dateRange = ref(null)
const filterClass = ref('')
const stats = ref({})

const pieChartRef = ref(null)
const barChartRef = ref(null)
const lineChartRef = ref(null)

let pieChart = null
let barChart = null
let lineChart = null

const COLOR_MAP = {
  crack: '#909399',
  spalling: '#409eff',
  efflorescence: '#e6a23c',
}
const LABEL_MAP = {
  crack: '裂缝',
  spalling: '剥落',
  efflorescence: '泛碱',
}

function classCount(name) {
  const item = (stats.value.by_class || []).find(c => c.class_name === name)
  return item?.count || 0
}

function formatTime(iso) {
  if (!iso) return '-'
  return iso.replace('T', ' ').slice(0, 19)
}

async function fetchRecords() {
  loading.value = true
  try {
    const { data } = await listRecords(page.value, pageSize)
    records.value = data.records
    total.value = data.total
  } catch {
    // ignore
  } finally {
    loading.value = false
  }
}

async function fetchStats() {
  const params = {}
  if (dateRange.value) {
    params.start_date = dateRange.value[0]
    params.end_date = dateRange.value[1]
  }
  if (filterClass.value) {
    params.class_name = filterClass.value
  }

  try {
    const { data } = await getStatsSummary(params)
    stats.value = data
    await nextTick()
    renderCharts(data)
  } catch {
    // ignore
  }
}

function renderCharts(data) {
  // 饼图
  if (pieChartRef.value) {
    if (!pieChart) pieChart = echarts.init(pieChartRef.value)
    const pieData = (data.by_class || []).map(c => ({
      name: LABEL_MAP[c.class_name] || c.class_name,
      value: c.count,
      itemStyle: { color: COLOR_MAP[c.class_name] || '#ccc' },
    }))
    pieChart.setOption({
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0 },
      series: [{
        type: 'pie',
        radius: ['40%', '65%'],
        label: { formatter: '{b}\n{c}' },
        data: pieData,
      }],
    })
  }

  // 柱状图
  if (barChartRef.value) {
    if (!barChart) barChart = echarts.init(barChartRef.value)
    const classes = (data.by_class || [])
    barChart.setOption({
      tooltip: {},
      xAxis: {
        type: 'category',
        data: classes.map(c => LABEL_MAP[c.class_name] || c.class_name),
      },
      yAxis: { type: 'value' },
      series: [{
        type: 'bar',
        data: classes.map(c => ({
          value: c.count,
          itemStyle: { color: COLOR_MAP[c.class_name] || '#409eff' },
        })),
        barWidth: '40%',
      }],
    })
  }

  // 折线图
  if (lineChartRef.value) {
    if (!lineChart) lineChart = echarts.init(lineChartRef.value)
    const dates = [...new Set((data.by_date_class || []).map(d => d.date))].sort()
    const classNames = [...new Set((data.by_date_class || []).map(d => d.class_name))]

    const series = classNames.map(cls => {
      const map = {}
      for (const row of data.by_date_class || []) {
        if (row.class_name === cls) map[row.date] = row.count
      }
      return {
        name: LABEL_MAP[cls] || cls,
        type: 'line',
        smooth: true,
        data: dates.map(d => map[d] || 0),
        itemStyle: { color: COLOR_MAP[cls] || '#409eff' },
      }
    })

    lineChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0 },
      xAxis: { type: 'category', data: dates },
      yAxis: { type: 'value' },
      series,
    })
  }
}

function viewRecord(id) {
  router.push({ path: '/', query: { recordId: id } })
}

async function handleDelete(id) {
  try {
    await deleteRecord(id)
    ElMessage.success('已删除')
    fetchRecords()
    fetchStats()
  } catch {
    ElMessage.error('删除失败')
  }
}

onMounted(() => {
  fetchRecords()
  fetchStats()
})

onBeforeUnmount(() => {
  pieChart?.dispose()
  barChart?.dispose()
  lineChart?.dispose()
})
</script>
