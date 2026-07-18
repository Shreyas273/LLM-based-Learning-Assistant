import React, { useEffect, useState } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { getProgress, getSessionReport } from '../../services/api';
import { motion } from 'framer-motion';
import { Loader2, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const MasterySection = () => {
    const userId = "guest";
    const [data, setData] = useState(null); // Combine progress and session report
    const [loading, setLoading] = useState(true);
    const [showResearch, setShowResearch] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [progressRes, reportRes] = await Promise.all([
                    getProgress(userId),
                    getSessionReport(userId)
                ]);
                setData({ progress: progressRes, report: reportRes });
            } catch (error) {
                console.error("Failed to load mastery data:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <div className="flex h-64 items-center justify-center"><Loader2 className="animate-spin text-sky-500" /></div>;
    if (!data) return <div className="text-center text-slate-400">Unable to load mastery data</div>;

    // Mastery Timeline Data
    const timelineData = data.report.difficulty_analysis?.difficulty_timeline || [];
    const timelineLabels = timelineData.map((t, i) => `Session ${i + 1}`);
    const timelineScores = timelineData.map(t => t.confidence || 0);

    const masteryGraphData = {
        labels: timelineLabels,
        datasets: [{
            label: 'Mastery Progression (BKT-est)',
            data: timelineScores,
            borderColor: 'rgba(16, 185, 129, 1)', // Emerald-500
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.3,
            fill: true,
            pointRadius: 4,
        }]
    };

    const graphOptions = {
        responsive: true,
        plugins: {
            legend: { labels: { color: '#94a3b8' } },
            tooltip: {
                callbacks: {
                    label: (context) => `Mastery: ${context.raw}%`
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(148, 163, 184, 0.1)' }
            },
            x: {
                ticks: { color: '#94a3b8' },
                grid: { display: false }
            }
        }
    };

    const curveOptions = {
        responsive: true,
        plugins: {
            legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(148,163,184,0.1)' }
            },
            x: {
                ticks: { color: '#94a3b8' },
                grid: { display: false }
            }
        }
    };

    // Dynamic Forgetting Curve
    // S = Base * avg_mastery_factor
    // avg_mastery = average confidence / 100
    const avgMastery = timelineScores.length > 0
        ? timelineScores.reduce((a, b) => a + b, 0) / timelineScores.length
        : 50;

    // S_estimated roughly maps 50% -> 5 days, 100% -> 30 days
    const estimatedStability = 1.0 + (avgMastery / 100.0) * 29.0;

    const forgettingDays = [0, 1, 3, 7, 14, 30];
    const dynamicRetention = forgettingDays.map(day => Math.exp(-day / estimatedStability) * 100);

    const curveData = {
        labels: forgettingDays.map(d => `Day ${d}`),
        datasets: [{
            label: `Dynamic Retention (S=${Math.round(estimatedStability)}d)`,
            data: dynamicRetention,
            borderColor: 'rgba(244, 63, 94, 1)', // Rose-500
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            tension: 0.4,
            pointRadius: 4,
        }]
    };

    return (
        <div className="space-y-6">
            {/* Research Context Header */}
            <div className="flex justify-between items-center">
                <h3 className="text-xl font-bold text-slate-100 flex items-center">
                    <TrendingUp className="mr-2 text-emerald-400" size={24} />
                    Learning Analytics
                </h3>
                <button
                    onClick={() => setShowResearch(!showResearch)}
                    className="text-xs text-slate-400 hover:text-sky-400 transition-colors border border-slate-700 rounded px-3 py-1 flex items-center"
                >
                    <AlertCircle size={12} className="mr-1" />
                    {showResearch ? "Hide Model Info" : "Research Context"}
                </button>
            </div>

            {/* Research Info Box */}
            {showResearch && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="bg-slate-900/50 border border-sky-500/20 rounded-lg p-4 text-sm text-slate-300"
                >
                    <h4 className="text-sky-400 font-semibold mb-2">Research-Level Metrics</h4>
                    <p className="mb-2">
                        This system utilizes a <span className="text-white">Hybrid Feature-Based Logistic Model</span> inspired by
                        <span className="text-white"> Bayesian Knowledge Tracing (BKT)</span> and
                        <span className="text-white"> Item Response Theory (IRT)</span>.
                    </p>
                    <ul className="list-disc list-inside space-y-1 text-slate-400 text-xs">
                        <li><strong>Dynamic Stability ($S$):</strong> Personalized forgetting curve based on mastery & difficulty ($S \propto M \times D$).</li>
                        <li><strong>Future Roadmap:</strong> Integration of Deep Knowledge Tracing (DKT) using LSTMs for temporal pattern recognition.</li>
                    </ul>
                </motion.div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Mastery Timeline */}
                <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="glass-panel p-6 rounded-2xl"
                >
                    <h3 className="text-lg font-bold text-slate-100 mb-6">Mastery Timeline</h3>
                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50 h-64">
                        {timelineScores.length > 0 ? (
                            <Line data={masteryGraphData} options={graphOptions} />
                        ) : (
                            <div className="flex h-full items-center justify-center text-slate-500">
                                Not enough data for timeline.
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* Dynamic Forgetting Curve */}
                <motion.div
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="glass-panel p-6 rounded-2xl"
                >
                    <h3 className="text-lg font-bold text-slate-100 mb-6 flex items-center justify-between">
                        <span>Retention Decay</span>
                        <span className="text-xs font-normal text-rose-400 bg-rose-500/10 px-2 py-1 rounded">
                            S ≈ {Math.round(estimatedStability)} days
                        </span>
                    </h3>
                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50 h-64">
                        <Line data={curveData} options={curveOptions} />
                    </div>
                </motion.div>
            </div>

            {/* Analysis Grid */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-4"
            >
                <div className="glass-card p-4 rounded-xl">
                    <h4 className="text-slate-400 text-sm mb-2">Strengths</h4>
                    {data.report.mastery_analysis?.strengths?.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                            {data.report.mastery_analysis.strengths.map((s, i) => (
                                <span key={i} className="text-xs bg-green-500/10 text-green-400 px-2 py-1 rounded border border-green-500/20">{s}</span>
                            ))}
                        </div>
                    ) : <p className="text-slate-600 text-xs">Keep practicing to identify strengths!</p>}
                </div>

                <div className="glass-card p-4 rounded-xl">
                    <h4 className="text-slate-400 text-sm mb-2">Needs Focus</h4>
                    {data.report.mastery_analysis?.areas_for_improvement?.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                            {data.report.mastery_analysis.areas_for_improvement.map((s, i) => (
                                <span key={i} className="text-xs bg-orange-500/10 text-orange-400 px-2 py-1 rounded border border-orange-500/20">{s}</span>
                            ))}
                        </div>
                    ) : <p className="text-slate-600 text-xs">Doing great! No immediate concerns.</p>}
                </div>

                <div className="glass-card p-4 rounded-xl">
                    <h4 className="text-slate-400 text-sm mb-2">Overall Trend</h4>
                    <div className="flex items-center space-x-2">
                        <CheckCircle2 size={16} className="text-sky-400" />
                        <span className="text-slate-200 capitalize">{data.report.learning_progress?.overall_trend?.replace('_', ' ') || 'Stable'}</span>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default MasterySection;
