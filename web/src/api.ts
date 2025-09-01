export type LoraSpec = { path: string; weight: number }
export type EmbeddingSpec = { path: string; token: string }

export type GenerateRequest = {
  modelPath: string
  prompt: string
  negativePrompt?: string
  width: number
  height: number
  steps: number
  cfg: number
  sampler: string
  seed?: number
  seedLocked?: boolean
  clipSkip?: number
  batchCount: number
  loras?: LoraSpec[]
  posEmbeddings?: EmbeddingSpec[]
  negEmbeddings?: EmbeddingSpec[]
}

export type ImageInfo = {
  id: string
  fileName: string
  width: number
  height: number
  seed: number
  sampler: string
  path: string
}

export type JobStatus = {
  status: 'queued' | 'running' | 'done' | 'error' | 'canceled'
  progress: number
  images: ImageInfo[]
  warnings?: string[]
  error?: string
}

const API_URL = (import.meta as any).env.VITE_API_URL || ''

export async function startGenerate(payload: GenerateRequest): Promise<{ jobId: string }> {
  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_URL}/job/${jobId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function imageUrl(imageId: string): string {
  return `${API_URL}/image/${imageId}`
}

export async function getLibraryIndex(): Promise<{ checkpoints: any[]; loras: any[]; embeddings: any[] }> {
  const res = await fetch(`${API_URL}/library/index`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

