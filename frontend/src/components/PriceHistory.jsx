import { useState, useEffect } from 'react'
import axios from 'axios'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function PriceHistory({ vegetable }) {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [range, setRange] = useState('all')

  useEffect(() => {
    if (!vegetable) return
    setLoading(true)
    axios.get(`/api/history/${encodeURIComponent(vegetable)}`)
      .then(res => setHistory(res.data.history || []))
      .finally(() => setLoading(false))
  }, [vegetable])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-emerald-600 border-t-transparent"></div>
        <span className="ml-3 text-gray-500 text-sm">Loading history...</span>
      </div>
    )
  }

  const now = new Date()
  const filtered = history.filter(p => {
    if (range === 'all') return true
    const d = new Date(p.date)
    const months = { '6m': 6, '1y': 12, '2y': 24, '5y': 60 }
    const cutoff = new Date(now)
    cutoff.setMonth(cutoff.getMonth() - months[range])
    return d >= cutoff
  })

  const prices = filtered.map(p => p.price)
  const avg = prices.length ? (prices.reduce((a, b) => a + b, 0) / prices.length) : 0
  const min = prices.length ? Math.min(...prices) : 0
  const max = prices.length ? Math.max(...prices) : 0
  const latest = prices.length ? prices[prices.length - 1] : 0

  const stats = [
    { label: 'Current Price', value: `Rs. ${latest.toFixed(0)}`, color: 'from-emerald-500 to-teal-500', textColor: 'text-white', icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
    )},
    { label: 'Average', value: `Rs. ${avg.toFixed(0)}`, color: 'from-blue-500 to-indigo-500', textColor: 'text-white', icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" /></svg>
    )},
    { label: 'Lowest', value: `Rs. ${min.toFixed(0)}`, color: 'from-cyan-500 to-teal-500', textColor: 'text-white', icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" /></svg>
    )},
    { label: 'Highest', value: `Rs. ${max.toFixed(0)}`, color: 'from-orange-500 to-red-500', textColor: 'text-white', icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
    )},
  ]

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map(s => (
          <div key={s.label} className={`relative overflow-hidden bg-gradient-to-br ${s.color} rounded-2xl shadow-lg p-5 ${s.textColor}`}>
            <div className="absolute top-0 right-0 w-20 h-20 bg-white/10 rounded-full -translate-y-6 translate-x-6"></div>
            <div className="relative">
              <div className="flex items-center gap-2 mb-2 opacity-80">
                {s.icon}
                <p className="text-xs font-semibold uppercase tracking-wider">{s.label}</p>
              </div>
              <p className="text-2xl sm:text-3xl font-bold">{s.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 p-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-5">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
            Price Trend - {vegetable}
          </h3>
          <div className="flex bg-gray-100 rounded-lg p-1 gap-0.5">
            {['6m', '1y', '2y', '5y', 'all'].map(r => (
              <button
                key={r}
                onClick={() => setRange(r)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all duration-200 ${
                  range === r
                    ? 'bg-white text-emerald-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {r.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={filtered} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              tickFormatter={d => d.slice(0, 7)}
              interval={Math.max(1, Math.floor(filtered.length / 12))}
            />
            <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} label={{ value: 'Price (LKR)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }} />
            <Tooltip
              formatter={(value) => [`Rs. ${value.toFixed(2)}`, 'Price']}
              labelFormatter={l => `Week: ${l}`}
              contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#priceGradient)"
              dot={false}
              activeDot={{ r: 5, fill: '#10b981', stroke: '#fff', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>

        <p className="text-xs text-gray-400 mt-3 flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {filtered.length} weekly data points | Source: Department of Census & Statistics, Sri Lanka
        </p>
      </div>
    </div>
  )
}
