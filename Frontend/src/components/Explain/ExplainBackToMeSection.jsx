import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MessageCircle, Send, Loader2, BookOpen, AlertCircle } from 'lucide-react';
import { explainConcept } from '../../services/api';

const ExplainBackToMeSection = () => {
    const userId = "guest";
    const [question, setQuestion] = useState('');
    const [studentAnswer, setStudentAnswer] = useState('');
    const [subject, setSubject] = useState('general');
    const [difficulty, setDifficulty] = useState('medium');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!question.trim() || !studentAnswer.trim()) {
            setError('Please enter both the question and your explanation.');
            return;
        }
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const response = await explainConcept({
                concept: question.trim(),
                explanation: studentAnswer.trim(),
                subject,
                difficulty,
                user_id: userId
            });
            setResult(response);
        } catch (err) {
            setError(err.message || 'Failed to analyze your explanation. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-panel p-6 rounded-2xl"
            >
                <div className="flex items-center gap-3 mb-6">
                    <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center border border-amber-500/30">
                        <MessageCircle className="h-6 w-6 text-amber-400" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-slate-100">Explain Back to Me</h2>
                        <p className="text-slate-400 text-sm">
                            Enter a concept question and explain it in your own words. Get feedback on misconceptions.
                        </p>
                    </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">Concept / Question</label>
                        <textarea
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            placeholder="e.g., What is the difference between TCP and UDP?"
                            rows={3}
                            className="w-full bg-slate-950/50 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 placeholder-slate-500 focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 outline-none resize-none transition-all"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">Your Explanation</label>
                        <textarea
                            value={studentAnswer}
                            onChange={(e) => setStudentAnswer(e.target.value)}
                            placeholder="Explain the concept in your own words..."
                            rows={5}
                            className="w-full bg-slate-950/50 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 placeholder-slate-500 focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 outline-none resize-none transition-all"
                        />
                    </div>
                    <div className="flex flex-wrap gap-4">
                        <div>
                            <label className="block text-xs text-slate-500 mb-1">Subject</label>
                            <select
                                value={subject}
                                onChange={(e) => setSubject(e.target.value)}
                                className="bg-slate-950/50 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:ring-2 focus:ring-amber-500/50 outline-none"
                            >
                                <option value="general">General</option>
                                <option value="cs">Computer Science</option>
                                <option value="math">Mathematics</option>
                                <option value="physics">Physics</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-xs text-slate-500 mb-1">Difficulty</label>
                            <select
                                value={difficulty}
                                onChange={(e) => setDifficulty(e.target.value)}
                                className="bg-slate-950/50 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:ring-2 focus:ring-amber-500/50 outline-none"
                            >
                                <option value="easy">Easy</option>
                                <option value="medium">Medium</option>
                                <option value="hard">Hard</option>
                            </select>
                        </div>
                    </div>
                    {error && (
                        <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                            <AlertCircle size={18} />
                            {error}
                        </div>
                    )}
                    <button
                        type="submit"
                        disabled={loading}
                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white font-semibold shadow-lg shadow-amber-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? (
                            <Loader2 className="animate-spin" size={20} />
                        ) : (
                            <Send size={20} />
                        )}
                        {loading ? 'Analyzing...' : 'Get Feedback'}
                    </button>
                </form>
            </motion.div>

            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel p-6 rounded-2xl"
                >
                    <h3 className="text-lg font-bold text-slate-100 mb-4 flex items-center gap-2">
                        <BookOpen size={20} className="text-amber-400" />
                        Misconception Analysis
                    </h3>
                    <div className="space-y-4">
                        {result.misconception_analysis && (() => {
                            let analysis = result.misconception_analysis;
                            if (typeof analysis === 'string') {
                                try { analysis = JSON.parse(analysis); } catch { return <p className="text-slate-300">{analysis}</p>; }
                            }
                            if (typeof analysis === 'string') return <p className="text-slate-300">{analysis}</p>;
                            return (
                                <div className="prose prose-invert prose-sm max-w-none prose-ai-response bg-slate-950/50 p-6 rounded-xl border border-slate-700/50">
                                    {analysis.raw_analysis ? (
                                        <p className="text-slate-300 whitespace-pre-wrap">{analysis.raw_analysis}</p>
                                    ) : (
                                        <div className="space-y-3">
                                            {analysis.correct_explanation && (
                                                <div>
                                                    <h4 className="font-semibold text-white mb-2">Correct Explanation</h4>
                                                    <p className="text-slate-300">{analysis.correct_explanation}</p>
                                                </div>
                                            )}
                                            {analysis.root_cause && (
                                                <div>
                                                    <h4 className="font-semibold text-amber-400 mb-2">Root Cause</h4>
                                                    <p className="text-slate-300">{analysis.root_cause}</p>
                                                </div>
                                            )}
                                            {analysis.improvement_suggestion && (
                                                <div>
                                                    <h4 className="font-semibold text-emerald-400 mb-2">Improvement Suggestion</h4>
                                                    <p className="text-slate-300">{analysis.improvement_suggestion}</p>
                                                </div>
                                            )}
                                            {analysis.targeted_example && (
                                                <div>
                                                    <h4 className="font-semibold text-sky-400 mb-2">Targeted Example</h4>
                                                    <p className="text-slate-300">{analysis.targeted_example}</p>
                                                </div>
                                            )}
                                            {analysis.detected_misconceptions && (
                                                <div>
                                                    <h4 className="font-semibold text-white mb-2">Detected Misconceptions</h4>
                                                    <ul className="list-disc list-inside space-y-1">
                                                        {(Array.isArray(analysis.detected_misconceptions) ? analysis.detected_misconceptions : []).map((m, i) => (
                                                            <li key={i} className="text-slate-300">{m}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                            {analysis.feedback && (
                                                <p className="text-slate-300">{analysis.feedback}</p>
                                            )}
                                            {analysis.correct_points && (
                                                <div>
                                                    <h4 className="font-semibold text-emerald-400 mb-2">What You Got Right</h4>
                                                    <ul className="list-disc list-inside space-y-1 text-slate-300">
                                                        {(Array.isArray(analysis.correct_points) ? analysis.correct_points : []).map((p, i) => (
                                                            <li key={i}>{p}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            );
                        })()}
                        {result.context_available && (
                            <p className="text-xs text-slate-500">Analysis used context from your uploaded documents.</p>
                        )}
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default ExplainBackToMeSection;
