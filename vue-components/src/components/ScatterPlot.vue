<script setup lang="ts">
import { ref, watch, onMounted, toRefs, computed, watchEffect } from 'vue'
import { ScatterGL } from 'scatter-gl'

import { createColorMap, linearScale } from '@colormap/core'
import { viridis, cividis, magma, inferno } from '@colormap/presets'
import type { ColorMap } from '@colormap/core'

import type { Vector3, Vector2 } from '../types'
import { toRGB } from '../utilities/colors'

const UNSELECTED_POINT_COLOR = 'rgba(189,189,189,255)'
const SELECTED_POINT_COLOR = 'rgba(70,70,70,255)'

type IdToPoint = Record<string, Vector3<number> | Vector2<number>>

interface Props {
  cameraPosition: number[]
  highlightedPoint: { id: string; is_transformed: boolean }
  points: IdToPoint
  transformedPoints: IdToPoint
  selectedPoints: string[]
}

const props = withDefaults(defineProps<Props>(), {
  cameraPosition: () => [],
  highlightedPoint: () => ({ id: '', is_transformed: false }),
  points: () => ({}),
  transformedPoints: () => ({}),
  selectedPoints: () => []
})

const { cameraPosition, points, transformedPoints, selectedPoints, highlightedPoint } =
  toRefs(props)

type Events = {
  cameraMove: [camera_position: Vector3<number> | Vector2<number>]
  hover: [{ id: string; is_transformed: boolean }]
  select: [ids: string[]]
}

const emit = defineEmits<Events>()

const plotContainer = ref<HTMLDivElement>()
const selectMode = ref<boolean>(true)
const colors = ref({ viridis, cividis, magma, inferno })
const colorMapName = ref<keyof typeof colors.value>('viridis')
const domain: Vector2<number> = [0, 1]
const range: Vector2<number> = [0, 1]
const scale = linearScale(domain, range)
const colorMapDomain = ref<Vector2<number>>(domain)
const colorMap = ref<ColorMap>(createColorMap(colors.value[colorMapName.value], scale))

let scatterPlot: ScatterGL | undefined
const scatterPlotRef = ref<ScatterGL>()
let hideSourcePoints = ref(false)

const sourcePointIds = computed(() => Object.keys(props.points))
const transformedPointIds = computed(() => Object.keys(props.transformedPoints))

const isTransformed = (index: number) => index >= sourcePointIds.value.length

const indexToId = (index: number) => {
  const ids = sourcePointIds.value
  if (index < ids.length) {
    return ids[index]
  } else {
    return transformedPointIds.value[index - ids.length]
  }
}

const idToIndex = (id: string, isTransformed: boolean) => {
  const ids = Object.keys(props.points)
  if (isTransformed) {
    return ids.indexOf(id)
  } else {
    return ids.length + Object.keys(props.transformedPoints).indexOf(id)
  }
}

const dataset = computed(() => {
  const allPoints = [...Object.values(points.value), ...Object.values(transformedPoints.value)]
  if (allPoints.length <= 0) {
    // If there are no points, we need to create a dataset with at least one point
    const ds = new ScatterGL.Dataset([[0, 0, 0]])
    // Then remove the point
    ds.points = []
    return ds
  }
  return new ScatterGL.Dataset(allPoints)
})

const sequences = computed(() => {
  return transformedPointIds.value.map((id) => ({
    indices: [idToIndex(id, false), idToIndex(id, true)]
  }))
})

watchEffect(() => {
  if (scatterPlotRef.value && scatterPlot) {
    // Due to a bug in scatter-gl we unselect all points before setting the sequences
    scatterPlot.select([])
    scatterPlot.setSequences(sequences.value)
  }
})

watch(
  [
    scatterPlotRef,
    dataset,
    selectedPoints,
    highlightedPoint,
    colorMap,
    hideSourcePoints,
    sequences
  ],
  () => scatterPlot?.render(dataset.value)
)

onMounted(() => {
  if (!plotContainer.value) {
    return
  }

  scatterPlot = new ScatterGL(plotContainer.value, {
    rotateOnStart: false,
    selectEnabled: true,
    pointColorer(i) {
      const id = indexToId(i)
      const isTrans = isTransformed(i)
      if (id === props.highlightedPoint.id && isTrans === props.highlightedPoint.is_transformed) {
        return `rgba(255,0,0,255)`
      }

      if (isTrans) {
        const p0 = props.points[id]
        const p1 = props.transformedPoints[id]
        const dx = Math.abs(p0[0] - p1[0])
        const dy = Math.abs(p0[1] - p1[1])
        const dz = p0.length == 3 && p1.length == 3 ? Math.abs(p0[2] - p1[2]) : 0
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
        const color = colorMap.value(dist)
        return `rgba(${toRGB(color)}, 255)`
      }

      if (props.selectedPoints.includes(id)) {
        // for original points that are selected
        return SELECTED_POINT_COLOR
      }

      if (hideSourcePoints.value) {
        return 'rgba(0,0,0,0)'
      }

      return UNSELECTED_POINT_COLOR
    },
    onHover(index: number | null) {
      const id = index == null ? '' : indexToId(index)
      const is_transformed = index == null ? false : isTransformed(index)
      emit('hover', { id, is_transformed })
    },
    onSelect(indices) {
      const ids = Array.from(new Set(indices.map(indexToId)))
      emit('select', ids)
    }
  })
  scatterPlotRef.value = scatterPlot

  const cameraControls = ((scatterPlot as any).scatterPlot as any).orbitCameraControls
  cameraControls.addEventListener('start', emitCameraPosition)
  cameraControls.addEventListener('change', emitCameraPosition)
  cameraControls.addEventListener('end', emitCameraPosition)
  updateCameraPosition(props.cameraPosition)

  // Without this there is an error upon browser refresh when sequences are defined.
  scatterPlot.render(dataset.value)
  // needs to be after render or pan mode and select are initially active at the same time
  scatterPlot.setSelectMode()
})

function emitCameraPosition() {
  if (scatterPlot) {
    let plotImpl = (scatterPlot as any).scatterPlot as any
    emit('cameraMove', plotImpl.camera.position.toArray())
  }
}

// Update the camera position if the prop changes
function updateCameraPosition(newValue: number[], oldValue: number[] = []) {
  // Only update if the parameters fit the precondition
  if (
    (newValue.length != 2 && newValue.length != 3) ||
    !newValue.every(function (v) {
      return v != null
    })
  ) {
    return
  }

  // Only update position if it is different, otherwise this can trigger an infinite loop
  if (
    (oldValue.length != 2 && oldValue.length != 3) ||
    !newValue.every(function (v, i) {
      v === oldValue[i]
    })
  ) {
    if (scatterPlot) {
      ;(scatterPlot as any).scatterPlot.setCameraPositionAndTarget(newValue, [0, 0, 0])
    }
  }
}

watch(cameraPosition, updateCameraPosition)

function updateColorMapDomain() {
  let bounds: Vector3<Vector2<number>> = [
    [Infinity, -Infinity],
    [Infinity, -Infinity],
    [Infinity, -Infinity]
  ]

  const points = Object.values(props.points)
  for (let i = 0; i < points.length; ++i) {
    for (let j = 0; j < 3; ++j) {
      const v = points[i][j] || 0
      if (v < bounds[j][0]) {
        bounds[j][0] = v
      }
      if (v > bounds[j][1]) {
        bounds[j][1] = v
      }
    }
  }

  const spans: Vector3<number> = [
    bounds[0][1] - bounds[0][0],
    bounds[1][1] - bounds[1][0],
    bounds[2][1] - bounds[2][0]
  ]

  colorMapDomain.value = [0, Math.max(...spans) / 3]
}

watch(points, () => {
  updateColorMapDomain()
})

function onSelectModeClick() {
  if (!selectMode.value) {
    selectMode.value = true
    scatterPlot?.setSelectMode()
  }
}

function onPanModeClick() {
  if (selectMode.value) {
    selectMode.value = false
    scatterPlot?.setPanMode()
  }
}

function onResetModeClick() {
  scatterPlot?.resetZoom()
  emitCameraPosition()
}

function onHideModeClick() {
  hideSourcePoints.value = !hideSourcePoints.value
}

function onColorMapChange(name: keyof typeof colors.value) {
  colorMapName.value = name
  const scale = linearScale(colorMapDomain.value, range)
  colorMap.value = createColorMap(colors.value[name], scale)
}

function onContainerResize() {
  if (scatterPlot) {
    scatterPlot.resize()
  }
}
</script>

<template>
  <q-resize-observer @resize="onContainerResize" :debounce="50" />
  <div style="width: 100%; height: 100%; position: relative; overflow: hidden">
    <div
      style="position: absolute; top: 0; left: 0; width: 100%; height: 100%"
      ref="plotContainer"
    ></div>
    <div style="position: absolute; top: 0; left: 0" class="q-pa-md q-gutter-sm">
      <q-toolbar classes="q-gutter-y-sm shadow-2">
        <q-btn
          class="q-ma-sm"
          round
          :color="selectMode ? 'white' : 'grey'"
          text-color="black"
          icon="mouse"
          @click="onPanModeClick"
        />
        <q-btn
          class="q-ma-sm"
          round
          :color="selectMode ? 'grey' : 'white'"
          text-color="black"
          icon="highlight_alt"
          @click="onSelectModeClick"
        />
        <q-btn
          class="q-ma-sm"
          round
          color="white"
          text-color="black"
          icon="refresh"
          @click="onResetModeClick"
        />
        <q-btn
          class="q-ma-sm"
          round
          :color="hideSourcePoints ? 'grey' : 'white'"
          text-color="black"
          icon="hide_source"
          @click="onHideModeClick"
        />
        <q-btn-dropdown class="q-ma-sm" rounded :label="colorMapName">
          <q-list>
            <q-item
              v-for="(_, key) in colors"
              clickable
              v-close-popup
              @click="() => onColorMapChange(key)"
              :key="key"
            >
              <q-item-section>
                <q-item-label>{{ key }}</q-item-label>
              </q-item-section>
            </q-item>
          </q-list>
        </q-btn-dropdown>
        <q-separator></q-separator>
      </q-toolbar>
    </div>
  </div>
</template>
