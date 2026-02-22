import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const STEPS = [
  { id: 1, title: 'Fetching Latest Data', desc: 'Downloading from statistics.gov.lk' },
  { id: 2, title: 'Preprocessing', desc: 'Cleaning & feature engineering' },
  { id: 3, title: 'Retraining Model', desc: 'XGBoost hyperparameter tuning' },
  { id: 4, title: 'Reloading', desc: 'Updating server with new model' },
]

export default function PipelineUpdater({ onClose }) {
  const [running, setRunning] = useState(false)
  const [steps, setSteps] = useState([])
  const [dataInfo, setDataInfo] = useState(null)
  const [finalMessage, setFinalMessage] = useState(null)
  const eventSourceRef = useRef(null)

  useEffect(() => {
    axios.get('/api/pipeline-status').then(res => setDataInfo(res.data))
    return () => { if (eventSourceRef.current) eventSourceRef.current.close() }
  }, [])

  const startPipeline = () => {
    setRunning(true)
    setSteps([])
    setFinalMessage(null)

    const es = new EventSource('/api/update-pipeline')
    eventSourceRef.current = es

    es.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.status === 'keepalive') return

      if (data.status === 'complete') {
        setFinalMessage({ type: 'success', message: data.message })
        setRunning(false)
        es.close()
        axios.get('/api/pipeline-status').then(res => setDataInfo(res.data))
        return
      }

      if (data.step === 0 && data.status === 'error') {
        setFinalMessage({ type: 'error', message: data.message })
        setRunning(false)
        es.close()
        return
      }

      setSteps(prev => {
        const idx = prev.findIndex(s => s.step === data.step)
        if (idx >= 0) {
          const updated = [...prev]
          updated[idx] = data
          return updated
        }
        return [...prev, data]
      })
    }

    es.onerror = () => {
      setFinalMessage({ type: 'error', message: 'Connection lost. Check if server is running.' })
      setRunning(false)
      es.close()
    }
  }

  const getStepStatus = (id) => steps.find(s => s.step === id)?.status || 'pending'
  const getStepMessage = (id) => steps.find(s => s.step === id)?.message || ''

  return (
    <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden animate-in z-50 relative" onClick={e => e.stopPropagation()}>
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-gray-50 to-white border-b border-gray-100">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center">
            <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 text-sm">Update Pipeline & Retrain</h3>
            {dataInfo && (
              <p className="text-xs text-gray-500">
                Current data: up to <span className="font-medium text-gray-700">{dataInfo.data_up_to}</span> | {dataInfo.total_records.toLocaleString()} records
              </p>
            )}
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-5">
        {/* Idle state */}
        {!running && !finalMessage && (
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <p className="text-sm text-gray-600">
                Fetch the latest prices from the government website, preprocess, and retrain the model with updated data.
              </p>
            </div>
            <button
              onClick={startPipeline}
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-medium rounded-xl transition-all duration-200 shadow-md shadow-emerald-600/20 hover:shadow-lg hover:shadow-emerald-600/30 text-sm whitespace-nowrap"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              </svg>
              Start Update
            </button>
          </div>
        )}

        {/* Progress */}
        {(running || finalMessage) && (
          <div className="space-y-2">
            {/* Steps as horizontal progress */}
            <div className="flex gap-2">
              {STEPS.map((step, i) => {
                const status = getStepStatus(step.id)
                const message = getStepMessage(step.id)
                return (
                  <div key={step.id} className="flex-1">
                    <div className={`rounded-xl p-3 border transition-all duration-300 ${
                      status === 'done'
                        ? 'bg-emerald-50 border-emerald-200'
                        : status === 'running'
                        ? 'bg-amber-50 border-amber-200 shadow-sm'
                        : status === 'error'
                        ? 'bg-red-50 border-red-200'
                        : 'bg-gray-50 border-gray-100'
                    }`}>
                      <div className="flex items-center gap-2 mb-1">
                        {status === 'done' ? (
                          <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
                            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        ) : status === 'running' ? (
                          <div className="w-5 h-5 rounded-full bg-amber-500 flex items-center justify-center animate-pulse">
                            <div className="w-2 h-2 rounded-full bg-white"></div>
                          </div>
                        ) : status === 'error' ? (
                          <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </div>
                        ) : (
                          <div className="w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center">
                            <span className="text-[10px] font-bold text-gray-400">{step.id}</span>
                          </div>
                        )}
                        <span className={`text-xs font-semibold ${
                          status === 'done' ? 'text-emerald-700'
                          : status === 'running' ? 'text-amber-700'
                          : status === 'error' ? 'text-red-700'
                          : 'text-gray-400'
                        }`}>{step.title}</span>
                      </div>
                      <p className="text-[10px] text-gray-500 leading-tight ml-7">
                        {message || step.desc}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Final message */}
            {finalMessage && (
              <div className={`flex items-center justify-between p-3 rounded-xl text-sm ${
                finalMessage.type === 'success'
                  ? 'bg-gradient-to-r from-emerald-50 to-teal-50 text-emerald-800 border border-emerald-200'
                  : 'bg-gradient-to-r from-red-50 to-pink-50 text-red-800 border border-red-200'
              }`}>
                <div className="flex items-center gap-2">
                  {finalMessage.type === 'success' ? (
                    <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                  <span>{finalMessage.message}</span>
                </div>
                <button onClick={() => { setSteps([]); setFinalMessage(null) }}
                  className="text-xs underline opacity-70 hover:opacity-100">Reset</button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
