import { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import PricePredictor from './components/PricePredictor'
import PriceHistory from './components/PriceHistory'
import ModelInsights from './components/ModelInsights'
import PipelineUpdater from './components/PipelineUpdater'

const TABS = [
  {
    key: 'Predict', label: 'Price Prediction', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    )
  },
  {
    key: 'History', label: 'Price History', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  },
  {
    key: 'Model Insights', label: 'Model Insights', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    )
  },
]

const CATEGORY_ICONS = {
  'Low Country Vegetables': (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  'Up Country Vegetables': (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
    </svg>
  ),
  'Leaves': (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
    </svg>
  ),
  'Potatoes': (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
    </svg>
  ),
}

const CATEGORY_COLORS = {
  'Low Country Vegetables': { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', active: 'bg-emerald-600', ring: 'ring-emerald-500', badge: 'bg-emerald-100 text-emerald-700', selectedBg: 'bg-emerald-500' },
  'Up Country Vegetables': { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700', active: 'bg-blue-600', ring: 'ring-blue-500', badge: 'bg-blue-100 text-blue-700', selectedBg: 'bg-blue-500' },
  'Leaves': { bg: 'bg-teal-50', border: 'border-teal-200', text: 'text-teal-700', active: 'bg-teal-600', ring: 'ring-teal-500', badge: 'bg-teal-100 text-teal-700', selectedBg: 'bg-teal-500' },
  'Potatoes': { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', active: 'bg-amber-600', ring: 'ring-amber-500', badge: 'bg-amber-100 text-amber-700', selectedBg: 'bg-amber-500' },
}

function App() {
  const [vegetables, setVegetables] = useState([])
  const [selectedVeg, setSelectedVeg] = useState('')
  const [activeTab, setActiveTab] = useState('Predict')
  const [showPipeline, setShowPipeline] = useState(false)
  const [pipelineBounce, setPipelineBounce] = useState(false)
  const [activeCategory, setActiveCategory] = useState('All')

  useEffect(() => {
    axios.get('/api/vegetables').then(res => {
      setVegetables(res.data.vegetables)
      if (res.data.vegetables.length > 0) {
        setSelectedVeg(res.data.vegetables[0].name)
      }
    })
  }, [])

  // Get unique categories
  const categories = useMemo(() => {
    const cats = [...new Set(vegetables.map(v => v.category))].filter(Boolean)
    return ['All', ...cats]
  }, [vegetables])

  // Filter vegetables by category
  const filteredVegetables = useMemo(() => {
    if (activeCategory === 'All') return vegetables
    return vegetables.filter(v => v.category === activeCategory)
  }, [vegetables, activeCategory])

  const selectedInfo = vegetables.find(v => v.name === selectedVeg)

  // Clean vegetable name for display (remove "1kg", "500g", "Bunch")
  const cleanName = (name) => {
    return name.replace(/\s+(1kg|1Kg|500g|Bunch)\s*$/i, '').trim()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-gray-50 to-emerald-50 flex flex-col">
      {/* Header */}
      <header className="relative">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-800 via-green-700 to-teal-700"></div>
          <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDM0djItSDI0di0yaDEyek0zNiAyNHYySDI0di0yaDEyeiIvPjwvZz48L2c+PC9zdmc+')] opacity-30"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-5">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl bg-white/15 backdrop-blur-sm flex items-center justify-center shadow-lg border border-white/20">
                <svg className="w-6 h-6 text-emerald-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                  VegPredict <span className="text-emerald-300 font-light">Sri Lanka</span>
                </h1>
                <p className="text-xs sm:text-sm text-emerald-200/80 hidden sm:block">
                  ML-powered vegetable price forecasting
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                id="pipeline-btn"
                onClick={() => setShowPipeline(!showPipeline)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 border ${pipelineBounce ? 'animate-bounce-attention' : ''
                  } ${showPipeline
                    ? 'bg-white text-emerald-700 border-white shadow-lg'
                    : 'bg-white/10 text-white border-white/20 hover:bg-white/20 backdrop-blur-sm'
                  }`}
                onAnimationEnd={() => setPipelineBounce(false)}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span className="hidden sm:inline">Update & Retrain</span>
              </button>

              <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 backdrop-blur-sm border border-white/20">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
                <span className="text-xs text-emerald-200 font-medium">XGBoost Model</span>
              </div>
            </div>
          </div>
        </div>

        {showPipeline && (
          <div className="relative">
            <div className="absolute top-0 right-0 left-0 z-50">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-4">
                <PipelineUpdater onClose={() => setShowPipeline(false)} />
              </div>
            </div>
          </div>
        )}
      </header>

      {showPipeline && (
        <div className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40" onClick={() => setShowPipeline(false)} />
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 relative z-10 flex-1 w-full">
        {/* Vegetable Selector with Category Filter */}
        <div className="mb-6 bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 overflow-hidden">
          {/* Category Tabs */}
          <div className="px-5 pt-5 pb-3">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Filter by Category
            </label>
            <div className="flex flex-wrap gap-2">
              {categories.map(cat => {
                const isActive = activeCategory === cat
                const colors = CATEGORY_COLORS[cat]
                const count = cat === 'All'
                  ? vegetables.length
                  : vegetables.filter(v => v.category === cat).length

                return (
                  <button
                    key={cat}
                    onClick={() => setActiveCategory(cat)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 border ${isActive
                        ? cat === 'All'
                          ? 'bg-gray-800 text-white border-gray-800 shadow-md shadow-gray-800/20'
                          : `${colors.active} text-white border-transparent shadow-md`
                        : cat === 'All'
                          ? 'bg-white text-gray-600 border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                          : `bg-white ${colors.text} ${colors.border} hover:${colors.bg}`
                      }`}
                  >
                    {cat !== 'All' && CATEGORY_ICONS[cat]}
                    <span>{cat}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded-md font-semibold ${isActive
                        ? 'bg-white/20 text-white'
                        : 'bg-gray-100 text-gray-500'
                      }`}>
                      {count}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Divider */}
          <div className="mx-5 border-t border-gray-100"></div>

          {/* Vegetable Cards */}
          <div className="px-5 pt-3 pb-5">
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
              Select Vegetable
              {activeCategory !== 'All' && (
                <span className="normal-case font-normal text-gray-400 ml-1">
                  — {activeCategory}
                </span>
              )}
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {filteredVegetables.map(v => {
                const isSelected = selectedVeg === v.name
                const colors = CATEGORY_COLORS[v.category] || CATEGORY_COLORS['Low Country Vegetables']

                return (
                  <button
                    key={v.name}
                    onClick={() => setSelectedVeg(v.name)}
                    className={`relative group text-left px-3 py-2.5 rounded-xl text-sm transition-all duration-200 border ${isSelected
                        ? `${colors.selectedBg} text-white border-transparent shadow-md`
                        : `bg-white ${colors.border} hover:shadow-sm hover:${colors.bg}`
                      }`}
                  >
                    <p className={`font-medium leading-tight text-xs ${isSelected ? 'text-white' : 'text-gray-800'
                      }`}>
                      {cleanName(v.name)}
                    </p>
                    <p className={`text-[10px] mt-0.5 ${isSelected ? 'text-white/70' : 'text-gray-400'
                      }`}>
                      {v.name.match(/1kg|1Kg|500g|Bunch/)?.[0] || ''}
                    </p>
                    {isSelected && (
                      <div className="absolute top-1 right-1.5">
                        <svg className="w-3.5 h-3.5 text-white/80" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="bg-white/60 backdrop-blur-sm rounded-xl p-1.5 shadow-sm border border-gray-200/60 inline-flex gap-1">
            {TABS.map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-2 px-4 sm:px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${activeTab === tab.key
                    ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/25'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-white'
                  }`}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
                <span className="sm:hidden">{tab.key === 'Model Insights' ? 'Insights' : tab.key}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="pb-8">
          {activeTab === 'Predict' && <PricePredictor vegetable={selectedVeg} onNeedUpdate={() => {
            window.scrollTo({ top: 0, behavior: 'smooth' })
            setTimeout(() => setPipelineBounce(true), 400)
          }} />}
          {activeTab === 'History' && <PriceHistory vegetable={selectedVeg} />}
          {activeTab === 'Model Insights' && <ModelInsights vegetable={selectedVeg} />}
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gradient-to-r from-gray-900 to-gray-800 border-t border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              <span>Data: Department of Census & Statistics, Sri Lanka</span>
            </div>
            <div className="flex items-center gap-4 text-gray-500 text-xs">
              <span className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                XGBoost Algorithm
              </span>
              <span>ML Assignment 2025</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
