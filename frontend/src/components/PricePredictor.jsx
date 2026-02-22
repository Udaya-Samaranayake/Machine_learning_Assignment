import { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

const WEEK_LABELS = ['1st Week', '2nd Week', '3rd Week', '4th Week']

// Generate next N months from base month/year
function getNextMonths(baseMonth, baseYear, count) {
  const months = []
  let m = baseMonth
  let y = baseYear
  for (let i = 0; i < count; i++) {
    m++
    if (m > 12) { m = 1; y++ }
    months.push({ month: m, year: y })
  }
  return months
}

export default function PricePredictor({ vegetable, onNeedUpdate }) {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingDefaults, setLoadingDefaults] = useState(false)
  const [showNotice, setShowNotice] = useState(false)

  const [latestMonth, setLatestMonth] = useState(null)
  const [latestYear, setLatestYear] = useState(null)
  const [latestPrice, setLatestPrice] = useState(null)

  // User selections
  const [selectedYear, setSelectedYear] = useState(null)
  const [selectedMonth, setSelectedMonth] = useState(null)
  const [selectedWeek, setSelectedWeek] = useState(1)

  // Available months (next 5 from latest data)
  const availableMonths = useMemo(() => {
    if (!latestMonth || !latestYear) return []
    return getNextMonths(latestMonth, latestYear, 5)
  }, [latestMonth, latestYear])

  // Get unique years from available months
  const availableYears = useMemo(() => {
    return [...new Set(availableMonths.map(m => m.year))]
  }, [availableMonths])

  // Get months for the selected year
  const monthsForYear = useMemo(() => {
    if (!selectedYear) return []
    return availableMonths.filter(m => m.year === selectedYear)
  }, [availableMonths, selectedYear])

  // Load defaults when vegetable changes
  useEffect(() => {
    if (!vegetable) return
    setLoadingDefaults(true)
    setPrediction(null)
    setSelectedYear(null)
    setSelectedMonth(null)
    setSelectedWeek(1)
    axios.get(`/api/defaults/${encodeURIComponent(vegetable)}`)
      .then(res => {
        const d = res.data
        setLatestMonth(d.month)
        setLatestYear(d.year)
        setLatestPrice(d.latest_price)
      })
      .finally(() => setLoadingDefaults(false))
  }, [vegetable])

  // Auto-select first year and month when available
  useEffect(() => {
    if (availableYears.length > 0 && !selectedYear) {
      setSelectedYear(availableYears[0])
    }
  }, [availableYears])

  useEffect(() => {
    if (monthsForYear.length > 0 && !selectedMonth) {
      setSelectedMonth(monthsForYear[0].month)
    }
  }, [monthsForYear])

  // When year changes, auto-select first valid month for that year
  const handleYearChange = (yr) => {
    setSelectedYear(yr)
    const months = availableMonths.filter(m => m.year === yr)
    if (months.length > 0) {
      setSelectedMonth(months[0].month)
    }
    setSelectedWeek(1)
    setPrediction(null)
  }

  const handleMonthChange = (m) => {
    setSelectedMonth(m)
    setSelectedWeek(1)
    setPrediction(null)
  }

  const handlePredict = () => {
    if (!vegetable || !selectedMonth || !selectedYear) return
    setLoading(true)
    axios.post('/api/predict', {
      vegetable,
      month: selectedMonth,
      year: selectedYear,
      week: selectedWeek,
    })
      .then(res => setPrediction(res.data))
      .finally(() => setLoading(false))
  }

  const isUp = prediction && prediction.price_change > 0
  const chartData = prediction ? [
    ...prediction.recent_prices.map(p => ({
      date: p.date.slice(5),
      price: p.price,
      type: 'actual',
    })),
    {
      date: 'Predicted',
      price: prediction.predicted_price,
      type: 'predicted',
    },
  ] : []

  return (
    <div className="space-y-6">
      {/* Input Form */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 overflow-hidden">
        <div className="px-6 py-4 bg-gradient-to-r from-emerald-600 to-teal-600">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Select Prediction Period
          </h3>
          <p className="text-emerald-100 text-xs mt-1">
            Choose a year, month, and week to forecast the price
            {latestMonth && latestYear && (
              <span className="ml-1">(data available up to {MONTH_NAMES[latestMonth - 1]} {latestYear})</span>
            )}
          </p>
        </div>

        <div className="p-6">
          {loadingDefaults ? (
            <div className="flex items-center justify-center py-6">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-emerald-600 border-t-transparent"></div>
              <span className="ml-3 text-gray-500 text-sm">Loading...</span>
            </div>
          ) : (
            <>
              {/* Step 1: Year */}
              <div className="mb-5">
                <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-100 text-emerald-700 text-[10px] font-bold mr-1.5">1</span>
                  Year
                </label>
                <div className="flex gap-2">
                  {availableYears.map(yr => (
                    <button
                      key={yr}
                      onClick={() => handleYearChange(yr)}
                      className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 border ${
                        selectedYear === yr
                          ? 'bg-gray-800 text-white border-gray-800 shadow-md shadow-gray-800/20'
                          : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      {yr}
                    </button>
                  ))}
                </div>
              </div>

              {/* Step 2: Month */}
              {selectedYear && (
                <div className="mb-5">
                  <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-100 text-emerald-700 text-[10px] font-bold mr-1.5">2</span>
                    Month
                  </label>
                  <div className="flex flex-wrap gap-2 items-center">
                    {monthsForYear.map(m => (
                      <button
                        key={m.month}
                        onClick={() => handleMonthChange(m.month)}
                        className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 border ${
                          selectedMonth === m.month
                            ? 'bg-emerald-600 text-white border-emerald-600 shadow-md shadow-emerald-600/20'
                            : 'bg-white text-gray-700 border-gray-200 hover:border-emerald-300 hover:bg-emerald-50'
                        }`}
                      >
                        {MONTH_NAMES[m.month - 1]}
                      </button>
                    ))}
                    {/* "More..." button */}
                    <button
                      onClick={() => setShowNotice(true)}
                      className="px-4 py-2.5 rounded-xl text-sm font-medium border border-dashed border-gray-300 text-gray-400 hover:text-gray-600 hover:border-gray-400 hover:bg-gray-50 transition-all duration-200"
                    >
                      more...
                    </button>
                  </div>
                </div>
              )}

              {/* Step 3: Week */}
              {selectedMonth && (
                <div className="mb-5">
                  <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                    <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-100 text-emerald-700 text-[10px] font-bold mr-1.5">3</span>
                    Week
                  </label>
                  <div className="flex gap-2">
                    {WEEK_LABELS.map((label, i) => (
                      <button
                        key={i + 1}
                        onClick={() => { setSelectedWeek(i + 1); setPrediction(null) }}
                        className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 border ${
                          selectedWeek === i + 1
                            ? 'bg-teal-600 text-white border-teal-600 shadow-md shadow-teal-600/20'
                            : 'bg-white text-gray-700 border-gray-200 hover:border-teal-300 hover:bg-teal-50'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Summary & Predict Button */}
              {selectedMonth && selectedYear && (
                <div className="flex items-center gap-4 pt-2">
                  <button
                    onClick={handlePredict}
                    disabled={loading}
                    className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-md shadow-emerald-600/20 hover:shadow-lg hover:shadow-emerald-600/30 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                        Predicting...
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Predict Price
                      </>
                    )}
                  </button>
                  <p className="text-sm text-gray-500">
                    Predicting for: <span className="font-semibold text-gray-800">{WEEK_LABELS[selectedWeek - 1]} of {MONTH_NAMES[selectedMonth - 1]} {selectedYear}</span>
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Notice Modal */}
      {showNotice && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={() => setShowNotice(false)}>
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm"></div>
          <div className="relative animate-modal bg-white rounded-2xl shadow-2xl border border-gray-200 max-w-md w-full p-6" onClick={e => e.stopPropagation()}>
            {/* Close button */}
            <button
              onClick={() => setShowNotice(false)}
              className="absolute top-3 right-3 p-1.5 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Icon */}
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-amber-100 flex items-center justify-center">
                <svg className="w-7 h-7 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>

            {/* Content */}
            <h3 className="text-lg font-bold text-gray-900 text-center mb-2">
              Prediction Limited to 5 Months
            </h3>
            <p className="text-sm text-gray-600 text-center mb-2">
              Predictions are available only for the <span className="font-semibold text-gray-800">next 5 months</span> from
              the latest dataset update
              {latestMonth && latestYear && (
                <span> (<span className="font-semibold">{MONTH_NAMES[latestMonth - 1]} {latestYear}</span>)</span>
              )}.
              Beyond this range, predictions may not be reliable.
            </p>
            <p className="text-sm text-gray-600 text-center mb-5">
              To predict further into the future, use the <span className="font-semibold text-emerald-700">Update & Retrain</span> button
              to fetch the latest data and retrain the model.
            </p>

            {/* Action button */}
            <div className="flex justify-center">
              <button
                onClick={() => {
                  setShowNotice(false)
                  if (onNeedUpdate) onNeedUpdate()
                }}
                className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-md shadow-emerald-600/20 hover:shadow-lg text-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Go to Update & Retrain
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {prediction && !loading && (
        <>
          {/* Prediction Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative overflow-hidden bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl shadow-lg shadow-emerald-600/20 p-6 text-white">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-8 translate-x-8"></div>
              <div className="relative">
                <p className="text-emerald-100 text-xs font-semibold uppercase tracking-wider">Predicted Price</p>
                <p className="text-4xl font-bold mt-2">
                  Rs. {prediction.predicted_price.toFixed(2)}
                </p>
                <p className="text-emerald-200 text-sm mt-2 flex items-center gap-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  {prediction.prediction_date}
                </p>
              </div>
            </div>

            <div className="relative overflow-hidden bg-white rounded-2xl shadow-sm border border-gray-200/60 p-6">
              <div className="absolute top-0 right-0 w-24 h-24 bg-blue-50 rounded-full -translate-y-6 translate-x-6"></div>
              <div className="relative">
                <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider">Last Known Price</p>
                <p className="text-4xl font-bold text-gray-900 mt-2">
                  Rs. {prediction.last_known_price.toFixed(2)}
                </p>
                <p className="text-sm text-gray-400 mt-2 flex items-center gap-1">
                  <span className="inline-block w-2 h-2 rounded-full bg-blue-400"></span>
                  {prediction.category}
                </p>
              </div>
            </div>

            <div className={`relative overflow-hidden rounded-2xl shadow-sm border p-6 ${
              isUp
                ? 'bg-gradient-to-br from-red-50 to-orange-50 border-red-200/60'
                : 'bg-gradient-to-br from-emerald-50 to-teal-50 border-emerald-200/60'
            }`}>
              <div className={`absolute top-0 right-0 w-24 h-24 rounded-full -translate-y-6 translate-x-6 ${isUp ? 'bg-red-100/50' : 'bg-emerald-100/50'}`}></div>
              <div className="relative">
                <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider">Expected Change</p>
                <div className="flex items-end gap-2 mt-2">
                  <p className={`text-4xl font-bold ${isUp ? 'text-red-600' : 'text-emerald-600'}`}>
                    {isUp ? '+' : ''}{prediction.pct_change.toFixed(2)}%
                  </p>
                  {isUp ? (
                    <svg className="w-6 h-6 text-red-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  ) : (
                    <svg className="w-6 h-6 text-emerald-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                    </svg>
                  )}
                </div>
                <p className={`text-sm mt-1 font-medium ${isUp ? 'text-red-500' : 'text-emerald-500'}`}>
                  {isUp ? '+' : ''}Rs. {prediction.price_change.toFixed(2)}
                </p>
              </div>
            </div>
          </div>

          {/* Selection Summary */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 p-5">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              Prediction Details
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Vegetable</p>
                <p className="font-semibold text-gray-800 mt-0.5">{prediction.vegetable}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Year</p>
                <p className="font-semibold text-gray-800 mt-0.5">{prediction.inputs_used.year}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Month</p>
                <p className="font-semibold text-gray-800 mt-0.5">{MONTH_NAMES[prediction.inputs_used.month - 1]}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Week</p>
                <p className="font-semibold text-gray-800 mt-0.5">{WEEK_LABELS[prediction.inputs_used.week - 1]}</p>
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-sm border border-gray-200/60 p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Recent Prices & Prediction
            </h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} label={{ value: 'Price (LKR)', angle: -90, position: 'insideLeft', style: { fontSize: 11, fill: '#9ca3af' } }} />
                <Tooltip
                  formatter={(value) => [`Rs. ${value.toFixed(2)}`, 'Price']}
                  contentStyle={{ borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
                />
                <Bar dataKey="price" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={index} fill={entry.type === 'predicted' ? '#f59e0b' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-6 mt-3 text-xs text-gray-500">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-sm bg-emerald-500"></div>
                Actual Prices
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-sm bg-amber-500"></div>
                Predicted
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
