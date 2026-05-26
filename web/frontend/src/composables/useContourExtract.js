/**
 * 掩码 → 轮廓提取（Marching Squares）+ 轮廓简化（Douglas-Peucker）
 * 用于画笔工具：将 canvas 像素掩码转为多边形顶点列表
 */

// Marching Squares 16 种情况的线段定义
// 每条线段由两个端点组成，端点是 cell 边的中点
// a=上中点, b=右中点, c=下中点, d=左中点
const SEGMENTS = {
  0: [],
  1: [[[0, 0.5], [0.5, 1]]],
  2: [[[0.5, 1], [1, 0.5]]],
  3: [[[0, 0.5], [1, 0.5]]],
  4: [[[0.5, 0], [1, 0.5]]],
  5: [[[0, 0.5], [0.5, 0]], [[0.5, 1], [1, 0.5]]],
  6: [[[0.5, 0], [0.5, 1]]],
  7: [[[0, 0.5], [0.5, 0]]],
  8: [[[0, 0.5], [0.5, 0]]],
  9: [[[0.5, 0], [0.5, 1]]],
  10: [[[0, 0.5], [0.5, 1]], [[0.5, 0], [1, 0.5]]],
  11: [[[0.5, 0], [1, 0.5]]],
  12: [[[0, 0.5], [1, 0.5]]],
  13: [[[0.5, 1], [1, 0.5]]],
  14: [[[0, 0.5], [0.5, 1]]],
  15: [],
}

/**
 * 从二值掩码提取轮廓点（Marching Squares）
 * @param {Uint8Array} mask - 二值掩码（0 或 1），行优先排列
 * @param {number} w - 掩码宽度
 * @param {number} h - 掩码高度
 * @returns {Array<[number,number]>} 轮廓顶点列表 [[x,y], ...]
 */
export function maskToContour(mask, w, h) {
  // 收集所有线段
  const segments = []
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const nw = mask[y * w + x] ? 1 : 0
      const ne = mask[y * w + x + 1] ? 1 : 0
      const sw = mask[(y + 1) * w + x] ? 1 : 0
      const se = mask[(y + 1) * w + x + 1] ? 1 : 0
      const caseIdx = nw * 8 + ne * 4 + se * 2 + sw
      const segs = SEGMENTS[caseIdx]
      if (!segs || segs.length === 0) continue
      for (const seg of segs) {
        segments.push([
          [x + seg[0][0], y + seg[0][1]],
          [x + seg[1][0], y + seg[1][1]],
        ])
      }
    }
  }

  if (segments.length === 0) return []

  // 链接线段为连续轮廓
  return chainSegments(segments)
}

/**
 * 将无序线段链为连续多边形顶点
 */
function chainSegments(segments) {
  if (segments.length === 0) return []

  const eps = 0.01
  const used = new Uint8Array(segments.length)
  const chains = []

  for (let i = 0; i < segments.length; i++) {
    if (used[i]) continue
    used[i] = 1
    const chain = [segments[i][0], segments[i][1]]

    let changed = true
    while (changed) {
      changed = false
      for (let j = 0; j < segments.length; j++) {
        if (used[j]) continue
        const [a, b] = segments[j]
        const tail = chain[chain.length - 1]
        const head = chain[0]

        if (dist(a, tail) < eps) {
          chain.push(b)
          used[j] = 1
          changed = true
        } else if (dist(b, tail) < eps) {
          chain.push(a)
          used[j] = 1
          changed = true
        } else if (dist(a, head) < eps) {
          chain.unshift(b)
          used[j] = 1
          changed = true
        } else if (dist(b, head) < eps) {
          chain.unshift(a)
          used[j] = 1
          changed = true
        }
      }
    }
    chains.push(chain)
  }

  // 取最长链作为主轮廓
  let longest = chains[0]
  for (const c of chains) {
    if (c.length > longest.length) longest = c
  }

  // 取整
  return longest.map(([x, y]) => [Math.round(x), Math.round(y)])
}

function dist(a, b) {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Douglas-Peucker 折线简化
 * @param {Array<[number,number]>} points - 输入顶点列表
 * @param {number} tolerance - 容差（像素），越大简化越多
 * @returns {Array<[number,number]>} 简化后的顶点列表
 */
export function simplifyContour(points, tolerance = 2) {
  if (points.length <= 2) return points

  // 找到距离首尾连线最远的点
  const first = points[0]
  const last = points[points.length - 1]
  let maxDist = 0
  let maxIdx = 0

  for (let i = 1; i < points.length - 1; i++) {
    const d = pointLineDist(points[i], first, last)
    if (d > maxDist) {
      maxDist = d
      maxIdx = i
    }
  }

  // 如果最远距离超过容差，递归简化
  if (maxDist > tolerance) {
    const left = simplifyContour(points.slice(0, maxIdx + 1), tolerance)
    const right = simplifyContour(points.slice(maxIdx), tolerance)
    return left.slice(0, -1).concat(right)
  }

  // 否则，用首尾两点代替所有中间点
  return [first, last]
}

/**
 * 点到线段的距离
 */
function pointLineDist(p, a, b) {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const lenSq = dx * dx + dy * dy
  if (lenSq === 0) return dist(p, a)

  let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lenSq
  t = Math.max(0, Math.min(1, t))

  const proj = [a[0] + t * dx, a[1] + t * dy]
  return dist(p, proj)
}
