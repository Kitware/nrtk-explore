<script setup lang="ts">
import { ref, computed, watch } from 'vue'

interface Props {
  currentDataset: string
  repositoryDatasets: string[]
  status: 'idle' | 'pending' | 'success' | 'fail'
  progress: number
}

type Events = {
  exportDataset: [{ name: string; full: boolean }]
}

const props = defineProps<Props>()

const emit = defineEmits<Events>()

function pathToFilename(path: string): string {
  return path.split('\\').pop()?.split('/').pop() || path
}

function filenameToBasename(filename: string) {
  const splitFilename = filename.split('.')

  if (splitFilename.length <= 1) {
    return filename
  } else {
    return splitFilename.slice(0, -1).join('.')
  }
}

const currentName = ref<string>(filenameToBasename(pathToFilename(props.currentDataset)))
const touched = ref<boolean>(false)
const overwrite = ref<boolean>(false)
const fullExport = ref<boolean>(false)

const datasetsNames = computed(() => {
  const datasetNames = props.repositoryDatasets.map(pathToFilename).map(filenameToBasename)
  return new Set(datasetNames)
})

const exporting = computed(() => {
  return props.status === 'pending'
})

const barColor = computed(() => {
  switch (props.status) {
    case 'success':
      return 'positive'
    case 'fail':
      return 'negative'
    case 'idle':
    case 'pending':
    default:
      return 'primary'
  }
})

const overwriteAlert = computed(() => {
  return (
    datasetsNames.value.has(currentName.value) &&
    !overwrite.value &&
    !(props.status === 'success' && !touched.value)
  )
})

watch(
  () => props.currentDataset,
  (currentDataset) => {
    currentName.value = filenameToBasename(pathToFilename(currentDataset))
    overwrite.value = false
    fullExport.value = false
    touched.value = false
  }
)

watch(
  () => props.repositoryDatasets,
  () => {
    overwrite.value = false
    fullExport.value = false
    touched.value = false
  }
)

function onOverwriteChange(value: boolean, _ev: MouseEvent) {
  overwrite.value = value
}

function onNameChange(value: string) {
  touched.value = true
  currentName.value = value
  overwrite.value = false
}

function onExport() {
  emit('exportDataset', { name: currentName.value, full: fullExport.value })
}
</script>

<template>
  <q-card-section>
    <q-input
      outlined
      dense
      stack-label
      :disable="exporting"
      :model-value="currentName"
      :error="overwriteAlert"
      error-message="Dataset with this name already exists. Overwrite it?"
      label="Exported Dataset Name"
      type="string"
      @update:model-value="onNameChange"
    />
    <div class="q-gutter-sm">
      <q-radio v-model="fullExport" :val="true" :disable="exporting" label="Full" />
      <q-radio v-model="fullExport" :val="false" :disable="exporting" label="Sampled" />
    </div>
    <q-toggle
      label="Overwrite"
      :model-value="overwrite"
      :disable="exporting || !datasetsNames.has(currentName)"
      @update:model-value="onOverwriteChange"
    />
    <q-field
      dense
      borderless
      no-error-icon
      hide-bottom-space
      :error="status === 'fail'"
      error-message="An error occurred while exporting."
    >
      <template v-slot:control>
        <q-linear-progress :value="progress" :color="barColor" class="" />
      </template>
    </q-field>
  </q-card-section>
  <q-card-actions align="right">
    <q-btn
      :disabled="exporting || (datasetsNames.has(currentName) && !overwrite)"
      @click="onExport"
      flat
    >
      Export
    </q-btn>
  </q-card-actions>
</template>
