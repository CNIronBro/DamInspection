import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

export const getModels = () => api.get('/models')

export const getConfig = () => api.get('/config')

export const detect = (formData) =>
  api.post('/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })

// 记录保存与查询
export const saveRecord = (data) => api.post('/records/save', data)
export const listRecords = (page = 1, size = 20) => api.get('/records', { params: { page, size } })
export const getRecord = (id) => api.get(`/records/${id}`)
export const deleteRecord = (id) => api.delete(`/records/${id}`)
export const getStatsSummary = (params) => api.get('/records/stats/summary', { params })

// 人工标注量化测量
export const measureAnnotation = (data) => api.post('/records/measure', data)
