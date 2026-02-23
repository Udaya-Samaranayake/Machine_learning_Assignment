import { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const PLOTS = [
  { key: 'shap_summary_bar.png', label: 'SHAP Importance', icon: '1' },
  { key: 'shap_beeswarm.png', label: 'SHAP Beeswarm', icon: '2' },
  { key: 'shap_dependence.png', label: 'SHAP Dependence', icon: '3' },
  { key: 'shap_waterfall.png', label: 'SHAP Waterfall', icon: '4' },
  { key: 'partial_dependence.png', label: 'Partial Dependence', icon: '5' },
  { key: 'actual_vs_predicted.png', label: 'Actual vs Predicted', icon: '6' },
]

const BAR_COLORS = [
  '#10b981', '#14b8a6', '#06b6d4', '#3b82f6', '#6366f1',
  '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e',
]

export default function ModelInsights({ vegetable }) {
  const [info, setInfo] = useState(null)
  const [activePlot, setActivePlot] = useState(PLOTS[0].key)
  const [loading, setLoading] = useState(true)
  const [plotLoading, setPlotLoading] = useState(false)
  const [plotKey, setPlotKey] = useState(0)

  useEffect(() => {
    axios.get('/api/model-info')
      .then(res => setInfo(res.data))
      .finally(() => setLoading(false))
  }, [])

  // When vegetable changes, show loading and pre-warm the cache
  useEffect(() => {
    if (!vegetable) return
    setPlotLoading(true)
    // Request the active plot to trigger generation; on load the img onLoad will clear loading
    setPlotKey(prev => prev + 1)
  }, [vegetable])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-emerald-600 border-t-transparent"></div>
        <span className="ml-3 text-gray-500 text-sm">Loading model info...</span>
      </div>
    )
  }

  if (!info) return null

  const topFeatures = info.feature_importance.slice(0, 10)

  const metrics = [
    {
      label: 'R² Score', value: info.test_r2.toFixed(4), subtitle: 'Variance Explained',
      color: 'from-emerald-500 to-teal-500',
      icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    },
    {
      label: 'RMSE', value: `${info.test_rmse.toFixed(2)}`, subtitle: 'LKR Error',
      color: 'from-blue-500 to-indigo-500',
      icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>,
    },
    {
      label: 'MAPE', value: `${info.test_mape.toFixed(2)}%`, subtitle: 'Avg % Error',
      color: 'from-amber-500 to-orange-500',
      icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>,
    },
    {
      label: 'Accuracy', value: `${(100 - info.test_mape).toFixed(1)}%`, subtitle: 'Overall',
      color: 'from-violet-500 to-purple-500',
      icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>,
    },
  ]

  return (
    <div className="space-y-6">
      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map(m => (
          <div key={m.label} className={`relative overflow-hidden bg-gradient-to-br ${m.color} rounded-2xl shadow-lg p-5 text-white`}>
            <div className="absolute top-0 right-0 w-20 h-20 bg-white/10 rounded-full -translate-y-6 translate-x-6"></div>
            <div className="relative">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-semibold uppercase tracking-wider opacity-80">{m.label}</p>
                <div className="opacity-60">{m.icon}</div>
              </div>
              <p className="text-3xl font-bold">{m.value}</p>
              <p className="text-xs mt-1 opacity-70">{m.subtitle}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Feature Importance Chart */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-5 flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Top 10 Feature Importance
        </h3>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={topFeatures} layout="vertical" margin={{ top: 5, right: 30, left: 140, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis type="number" tick={{ fontSize: 11, fill: '#9ca3af' }} />
            <YAxis dataKey="feature" type="category" tick={{ fontSize: 11, fill: '#6b7280' }} width={140} />
            <Tooltip
              formatter={(value) => [value.toFixed(4), 'Importance']}
              contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
            />
            <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
              {topFeatures.map((entry, index) => (
                <Cell key={index} fill={BAR_COLORS[index % BAR_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Explainability Plots */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-gray-50 to-white border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Explainability Visualizations{vegetable ? ` — ${vegetable.replace(/\s+(1kg|1Kg|500g|Bunch)\s*$/i, '').trim()}` : ' (SHAP & PDP)'}
          </h3>
        </div>

        <div className="p-6">
          {/* Plot selector */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 mb-5">
            {PLOTS.map(plot => (
              <button
                key={plot.key}
                onClick={() => { setActivePlot(plot.key); setPlotLoading(true); setPlotKey(prev => prev + 1) }}
                className={`px-3 py-2.5 text-xs font-semibold rounded-xl transition-all duration-200 border ${activePlot === plot.key
                  ? 'bg-emerald-600 text-white border-emerald-600 shadow-md shadow-emerald-600/20'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-emerald-300 hover:text-emerald-700'
                  }`}
              >
                {plot.label}
              </button>
            ))}
          </div>

          {/* Plot image */}
          <div className="relative border border-gray-200 rounded-xl overflow-hidden bg-white shadow-inner min-h-[200px]">
            {plotLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-10">
                <div className="flex flex-col items-center gap-2">
                  <div className="animate-spin rounded-full h-8 w-8 border-2 border-emerald-600 border-t-transparent"></div>
                  <span className="text-sm text-gray-500">Generating plot for selected vegetable...</span>
                </div>
              </div>
            )}
            <img
              key={`${vegetable}-${activePlot}-${plotKey}`}
              src={vegetable ? `/api/vegetable-plots/${encodeURIComponent(vegetable)}/${activePlot}` : `/api/plots/${activePlot}`}
              alt={activePlot}
              className="w-full h-auto"
              onLoad={() => setPlotLoading(false)}
              onError={() => setPlotLoading(false)}
            />
          </div>
        </div>
      </div>

    </div>
  )
}
