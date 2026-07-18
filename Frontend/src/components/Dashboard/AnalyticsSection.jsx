import React, { useEffect, useState } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { getSessionReport } from '../../services/api';
import { motion } from 'framer-motion';
import { Loader2, Activity, Clock, Award, BookOpen, Sparkles, ListChecks } from 'lucide-react';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const AnalyticsSection = () => {
    const userId = "guest";
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                const response = await getSessionReport(userId);
                setReport(response);
            } catch (error) {
                console.error("Failed to load analytics:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, []);

    if (loading) {
        return (
            <div className="flex h-64 items-center justify-center">
                <Loader2 className="animate-spin text-sky-500" size={32} />
            </div>
        );
    }

    if (!report) return <div className="text-center text-slate-400">No data available</div>;

    const topics = report.summary?.topic_wise_distribution ? Object.keys(report.summary.topic_wise_distribution) : [];
    const counts = report.summary?.topic_wise_distribution ? Object.values(report.summary.topic_wise_distribution) : [];

    const chartData = {
        labels: topics,
        datasets: [
            {
                label: 'Questions per Topic',
                data: counts,
                backgroundColor: 'rgba(56, 189, 248, 0.5)',
                borderColor: 'rgba(56, 189, 248, 1)',
                borderWidth: 1,
                borderRadius: 8,
            },
        ],
    };


    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
                labels: { color: '#94a3b8' }
            },
            title: {
                display: false,
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(148, 163, 184, 0.1)' }
            },
            x: {
                ticks: { color: '#94a3b8' },
                grid: { display: false }
            }
        }
    };

    const learningAnalytics = report.learning_analytics || null;

    return (
        <div className="space-y-6">
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
            >
                <div className="glass-card p-4 rounded-xl flex items-center space-x-4">
                    <div className="p-3 bg-blue-500/10 rounded-lg text-blue-400">
                        <Activity size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-400">Total Questions</p>
                        <p className="text-2xl font-bold text-slate-100">{report.summary.total_questions_asked}</p>
                    </div>
                </div>

                <div className="glass-card p-4 rounded-xl flex items-center space-x-4">
                    <div className="p-3 bg-green-500/10 rounded-lg text-green-400">
                        <Award size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-400">Avg Confidence</p>
                        <p className="text-2xl font-bold text-slate-100">{Math.round(report.summary.average_confidence)}%</p>
                    </div>
                </div>

                <div className="glass-card p-4 rounded-xl flex items-center space-x-4">
                    <div className="p-3 bg-purple-500/10 rounded-lg text-purple-400">
                        <BookOpen size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-400">Topics Covered</p>
                        <p className="text-2xl font-bold text-slate-100">{topics.length}</p>
                    </div>
                </div>

                <div className="glass-card p-4 rounded-xl flex items-center space-x-4">
                    <div className="p-3 bg-orange-500/10 rounded-lg text-orange-400">
                        <Clock size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-slate-400">Study Time</p>
                        <p className="text-xl font-bold text-slate-100">{report.engagement_metrics?.estimated_session_duration_minutes || 0} min</p>
                    </div>
                </div>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
                <motion.div
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-panel p-6 rounded-2xl"
                >
                    <h3 className="text-lg font-semibold text-slate-200 mb-4">Topic Distribution</h3>
                    <Bar data={chartData} options={chartOptions} />
                </motion.div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 }}
                    className="glass-panel p-6 rounded-2xl"
                >
                    <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                        <Sparkles size={18} className="text-sky-400" />
                        Learning Insights
                    </h3>

                    {learningAnalytics ? (
                        <div className="space-y-4">
                            <div className="flex items-center justify-between bg-slate-900/40 border border-slate-700/40 rounded-xl p-3">
                                <div className="text-sm text-slate-400">Learning style</div>
                                <div className="text-sm font-semibold text-slate-100">{learningAnalytics.learning_style || '—'}</div>
                            </div>

                            <div>
                                <div className="text-xs text-slate-500 uppercase mb-2">Weak topics</div>
                                {Array.isArray(learningAnalytics.weak_topics) && learningAnalytics.weak_topics.length > 0 ? (
                                    <div className="space-y-2">
                                        {learningAnalytics.weak_topics.slice(0, 6).map((w, idx) => (
                                            <div key={idx} className="flex items-center justify-between bg-slate-900/40 border border-slate-700/40 rounded-lg px-3 py-2">
                                                <div className="text-sm text-slate-200">{w.topic}</div>
                                                <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded">
                                                    {Math.round(Number(w.weakness_score || 0))}%
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-sm text-slate-500">No weak topics detected yet.</div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="text-sm text-slate-500">
                            Learning analytics will appear after you’ve asked a few questions.
                        </div>
                    )}
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-panel p-6 rounded-2xl"
                >
                    <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
                        <ListChecks size={18} className="text-emerald-400" />
                        Weekly Study Plan
                    </h3>

                    {learningAnalytics?.weekly_plan?.length ? (
                        <div className="space-y-2">
                            {learningAnalytics.weekly_plan.map((d, idx) => (
                                <div key={idx} className="bg-slate-900/40 border border-slate-700/40 rounded-xl p-3">
                                    <div className="flex items-center justify-between">
                                        <div className="text-sm font-semibold text-slate-100">{d.day}</div>
                                        <div className="text-[11px] text-slate-400">{d.estimated_time || '15 minutes'}</div>
                                    </div>
                                    <div className="text-sm text-slate-300 mt-1">
                                        Focus: <span className="text-slate-100">{d.focus_topic}</span>
                                    </div>
                                    <div className="text-[11px] text-slate-500 mt-1">
                                        {d.urgency ? `Urgency: ${d.urgency}` : null}
                                    </div>
                                    {Array.isArray(d.activities) && d.activities.length > 0 && (
                                        <ul className="mt-2 text-xs text-slate-400 list-disc list-inside space-y-0.5">
                                            {d.activities.slice(0, 3).map((a, i) => <li key={i}>{a}</li>)}
                                        </ul>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-sm text-slate-500">No weekly plan available yet.</div>
                    )}
                </motion.div>
            </div>
        </div>
    );
};

export default AnalyticsSection;
