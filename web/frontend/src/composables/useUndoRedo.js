import { ref, computed } from 'vue'

export function useUndoRedo(maxSize = 50) {
  const stack = ref([])
  const pointer = ref(-1)

  const canUndo = computed(() => pointer.value >= 0)
  const canRedo = computed(() => pointer.value < stack.value.length - 1)

  function push(state) {
    // 丢弃 pointer 之后的所有记录
    stack.value = stack.value.slice(0, pointer.value + 1)
    stack.value.push(JSON.parse(JSON.stringify(state)))
    if (stack.value.length > maxSize) {
      stack.value.shift()
    } else {
      pointer.value++
    }
  }

  function undo() {
    if (pointer.value < 0 || pointer.value >= stack.value.length) return null
    pointer.value--
    if (pointer.value < 0) return null
    return JSON.parse(JSON.stringify(stack.value[pointer.value]))
  }

  function redo() {
    if (pointer.value >= stack.value.length - 1) return null
    pointer.value++
    return JSON.parse(JSON.stringify(stack.value[pointer.value]))
  }

  function clear() {
    stack.value = []
    pointer.value = -1
  }

  return { push, undo, redo, clear, canUndo, canRedo }
}
