import React, { useEffect, useMemo, useRef, useState } from 'react'
import { GenerateRequest, JobStatus, startGenerate, getJob, imageUrl, getLibraryIndex } from './api'

const balancedSamplers = ['Euler a', 'Euler', 'DPM++ SDE', 'DPM++ 2M', 'UniPC']

type SizePreset = 'Portrait' | 'Landscape' | 'Square' | 'Custom'

function usePoll<T>(fn: () => Promise<T>, deps: any[], ms = 500, active = true) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!active) return
    let stop = false
    let timer: any
    async function tick() {
      try {
        const v = await fn()
        if (!stop) setData(v)
      } catch (e: any) {
        if (!stop) setError(e?.message || String(e))
      } finally {
        if (!stop) timer = setTimeout(tick, ms)
      }
    }
    tick()
    return () => { stop = true; if (timer) clearTimeout(timer) }
  }, [...deps, active])
  return { data, error }
}

export default function App() {
  const [prompt, setPrompt] = useState('a photorealistic calico cat wearing aviator goggles, soft lighting, 85mm lens')
  const [showNeg, setShowNeg] = useState(false)
  const [negPrompt, setNegPrompt] = useState('blurry, low quality, lowres')
  const [cfg, setCfg] = useState(7.5)
  const [steps, setSteps] = useState(30)
  const [sampler, setSampler] = useState('Euler a')
  const [seed, setSeed] = useState<number | ''>('' as any)
  const [seedLocked, setSeedLocked] = useState(false)
  const [clipSkip, setClipSkip] = useState(0)
  const [sizePreset, setSizePreset] = useState<SizePreset>('Square')
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [batchCount, setBatchCount] = useState(2)
  const [modelPath, setModelPath] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const [library, setLibrary] = useState<{ checkpoints: any[] }>({ checkpoints: [] })
  const [loadingLibrary, setLoadingLibrary] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const lastProgressRef = useRef<number>(0)
  const timerRef = useRef<any>(null)

  useEffect(() => {
    if (sizePreset === 'Portrait') { setWidth(832); setHeight(1216) }
    else if (sizePreset === 'Landscape') { setWidth(1216); setHeight(832) }
    else if (sizePreset === 'Square') { setWidth(1024); setHeight(1024) }
  }, [sizePreset])

  const [pollActive, setPollActive] = useState<boolean>(false)
  useEffect(() => { setPollActive(!!jobId) }, [jobId])
  const { data: job } = usePoll<JobStatus>(
    async () => jobId ? await getJob(jobId) : Promise.resolve({ status: 'queued', progress: 0, images: [] } as any),
    [jobId],
    500,
    pollActive
  )

  useEffect(() => {
    if (job?.images?.length) {
      setSelectedImageId(job.images[job.images.length - 1].id)
    }
  }, [job?.images?.length])

  // Log when a new image arrives
  const lastImgCountRef = useRef<number>(0)
  useEffect(() => {
    const n = job?.images?.length ?? 0
    if (n > lastImgCountRef.current) {
      setLogs(prev => [...prev, `Image ${n} received`])
      lastImgCountRef.current = n
    }
  }, [job?.images?.length])

  // Timer handling
  useEffect(() => {
    if (startTime != null && (job?.status === 'running' || (jobId && !job))) {
      if (!timerRef.current) {
        timerRef.current = setInterval(() => {
          setElapsedMs(Date.now() - (startTime as number))
        }, 100)
      }
    } else {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
    }
    return () => { if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null } }
  }, [job?.status, startTime, jobId])

  // Stop polling and finalize timer when job finishes
  useEffect(() => {
    if (job && ['done','error','canceled'].includes(job.status as any)) {
      setPollActive(false)
      if (startTime != null) {
        setElapsedMs(Date.now() - startTime)
        setStartTime(null)
        setLogs(prev => [...prev, `Job ${job.status} in ${(Date.now()-startTime).toFixed(0)} ms`])
      }
    }
  }, [job?.status])

  async function onGenerate() {
    if (!modelPath) {
      alert('Choose a model path from the library')
      return
    }
    const payload: GenerateRequest = {
      modelPath,
      prompt,
      negativePrompt: showNeg ? negPrompt : '',
      width, height, steps, cfg, sampler,
      seed: seed === '' ? undefined : Number(seed),
      seedLocked,
      clipSkip,
      batchCount,
      loras: [],
      posEmbeddings: [],
      negEmbeddings: [],
    }
    setLogs(prev => [
      ...prev,
      `Generate clicked: ${new Date().toLocaleTimeString()}`,
      `Model: ${modelPath.split('\\').pop() || modelPath}`,
      `Size: ${width}x${height}, Steps: ${steps}, CFG: ${cfg}, Sampler: ${sampler}`,
    ])
    setStartTime(Date.now())
    setElapsedMs(0)
    lastProgressRef.current = 0
    const { jobId } = await startGenerate(payload)
    setJobId(jobId)
    setPollActive(true)
    setLogs(prev => [...prev, `Job started: ${jobId}`])
  }

  async function loadLibrary() {
    setLoadingLibrary(true)
    try { setLibrary(await getLibraryIndex() as any) }
    catch (e: any) { alert(e?.message || String(e)) }
    finally { setLoadingLibrary(false) }
  }

  useEffect(() => { loadLibrary() }, [])

  const selectedUrl = useMemo(() => selectedImageId ? imageUrl(selectedImageId) : '', [selectedImageId])

  // Log progress changes (every 5%) and warnings
  useEffect(() => {
    if (!job) return
    const p = job.progress ?? 0
    const last = lastProgressRef.current
    if (p >= last + 5) {
      lastProgressRef.current = p
      setLogs(prev => [...prev.slice(-500), `Progress: ${p}%`])
    }
    if (job.warnings && job.warnings.length) {
      setLogs(prev => {
        const existing = new Set(prev)
        const newItems = job.warnings.filter(w => !existing.has(w))
        return newItems.length ? [...prev, ...newItems] : prev
      })
    }
  }, [job?.progress, job?.warnings?.length])

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 16, padding: 16 }}>
      <div>
        <h1 style={{ marginTop: 0 }}>Generate</h1>
        <section>
          <label style={{ display: 'block', fontWeight: 600 }}>Prompt</label>
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} style={{ width: '100%' }} />
          <div style={{ marginTop: 6 }}>
            {!showNeg && <button onClick={() => setShowNeg(true)}>Add negative prompt</button>}
          </div>
          {showNeg && (
            <div style={{ marginTop: 8 }}>
              <label style={{ display: 'block', fontWeight: 600 }}>Negative prompt</label>
              <input value={negPrompt} onChange={e => setNegPrompt(e.target.value)} style={{ width: '100%' }} />
            </div>
          )}
        </section>

        <section style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginTop: 16 }}>
          <div>
            <label><b>CFG</b> {cfg.toFixed(1)}</label>
            <input type="range" min={0} max={20} step={0.1} value={cfg} onChange={e => setCfg(parseFloat(e.target.value))} />
          </div>
          <div>
            <label><b>Steps</b> {steps}</label>
            <input type="range" min={1} max={200} step={1} value={steps} onChange={e => setSteps(parseInt(e.target.value))} />
          </div>
          <div>
            <label><b>Sampler</b></label>
            <select value={sampler} onChange={e => setSampler(e.target.value)} style={{ width: '100%' }}>
              <optgroup label="Balanced">
                {balancedSamplers.map(s => <option key={s} value={s}>{s}</option>)}
              </optgroup>
            </select>
          </div>
          <div>
            <label><b>Seed</b></label>
            <input placeholder="random" value={seed} onChange={e => setSeed(e.target.value === '' ? '' : Number(e.target.value))} style={{ width: '100%' }} />
            <label style={{ display: 'block', marginTop: 4 }}>
              <input type="checkbox" checked={seedLocked} onChange={e => setSeedLocked(e.target.checked)} /> lock seed
            </label>
          </div>
          <div>
            <label><b>CLIP skip</b></label>
            <select value={clipSkip} onChange={e => setClipSkip(parseInt(e.target.value))} style={{ width: '100%' }}>
              {[0,1,2,3].map(x => <option key={x} value={x}>{x === 0 ? 'Off' : x}</option>)}
            </select>
          </div>
          <div>
            <label><b>Batch</b></label>
            <input type="number" min={1} max={8} value={batchCount} onChange={e => setBatchCount(parseInt(e.target.value))} style={{ width: '100%' }} />
          </div>
        </section>

        <section style={{ marginTop: 16 }}>
          <label style={{ display: 'block', fontWeight: 600 }}>Size & Orientation</label>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {(['Portrait','Landscape','Square','Custom'] as SizePreset[]).map(p => (
              <button key={p} onClick={() => setSizePreset(p)} style={{ padding: '6px 10px', borderRadius: 20, border: '1px solid #ccc', background: sizePreset===p? '#111': '#fff', color: sizePreset===p? '#fff': '#111' }}>{p}</button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <input type="number" min={64} max={2048} value={width} disabled={sizePreset!=='Custom'} onChange={e => setWidth(parseInt(e.target.value))} />
            <input type="number" min={64} max={2048} value={height} disabled={sizePreset!=='Custom'} onChange={e => setHeight(parseInt(e.target.value))} />
          </div>
        </section>

        <section style={{ marginTop: 16 }}>
          <label style={{ display: 'block', fontWeight: 600 }}>Model</label>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select value={modelPath} onChange={e => setModelPath(e.target.value)} style={{ flex: 1 }}>
              <option value="">-- Choose checkpoint --</option>
              {library.checkpoints
                .filter((c: any) => typeof c?.path === 'string' && c.path.toLowerCase().endsWith('.safetensors'))
                .map((c: any) => <option key={c.path} value={c.path}>{c.name}</option>)}
            </select>
            <button onClick={loadLibrary} disabled={loadingLibrary}>{loadingLibrary ? 'Scanning...' : 'Rescan'}</button>
          </div>
        </section>

        <div style={{ marginTop: 16 }}>
          <button onClick={onGenerate} disabled={job?.status === 'running'} style={{ padding: '10px 16px', fontWeight: 600 }}>
            {job?.status === 'running' ? `Generating… ${job?.progress ?? 0}%` : 'Generate'}
          </button>
        </div>
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <progress value={job?.status === 'running' ? job?.progress ?? 0 : 0} max={100} style={{ width: 260, height: 14 }} />
            <span style={{ minWidth: 70, display: 'inline-block', textAlign: 'right' }}>{job?.status === 'running' ? `${job?.progress ?? 0}%` : '0%'}</span>
            <span style={{ marginLeft: 12, fontVariantNumeric: 'tabular-nums' }}>
              ⏱ {((startTime ? elapsedMs : elapsedMs) / 1000).toFixed(1)}s
            </span>
          </div>
          <div style={{ marginTop: 8 }}>
            <label style={{ display: 'block', fontWeight: 600 }}>Logs</label>
            <textarea readOnly value={logs.join('\n')} rows={6} style={{ width: '100%', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' }} />
          </div>
          {job?.warnings?.length ? (
            <div style={{ color: '#a78400', marginTop: 6 }}>{job.warnings.join(' • ')}</div>
          ) : null}
        </div>
      </div>

      <div>
        <h2 style={{ marginTop: 0 }}>Viewer</h2>
        <div style={{ border: '1px solid #ddd', background: '#fff', minHeight: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {selectedUrl ? (
            <img src={selectedUrl} style={{ maxWidth: '100%', maxHeight: 480, objectFit: 'contain' }} />
          ) : (
            <div style={{ color: '#666' }}>No image yet</div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8, overflowX: 'auto', marginTop: 8 }}>
          {job?.images?.map(img => (
            <img key={img.id} src={imageUrl(img.id)} onClick={() => setSelectedImageId(img.id)} style={{ height: 96, border: selectedImageId===img.id? '2px solid #111':'1px solid #ccc', cursor: 'pointer' }} />
          ))}
        </div>
      </div>
    </div>
  )
}
