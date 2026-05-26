import { createRouter, createWebHistory } from 'vue-router'
import DetectionView from '../views/DetectionView.vue'

const routes = [
  { path: '/', name: 'detection', component: DetectionView },
  {
    path: '/stats',
    name: 'stats',
    component: () => import('../views/StatsView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
