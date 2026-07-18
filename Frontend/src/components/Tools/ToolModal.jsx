import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSanitize from 'rehype-sanitize';
import { X, Upload, Loader2, Sparkles } from 'lucide-react';
import { summarizeFile, solveImageProblem, generateContent, solveProblem } from '../../services/api';

const ToolModal = ({ tool, onClose }) => {
    const [input, setInput] = useState('');
    const [file, setFile] = useState(null);
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    if (!tool) return null;

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult('');

        try {
            let response;

            // 1. File Upload / Image Analysis
            if (file) {
                const formData = new FormData();
                formData.append('file', file);

                if (tool.type === 'summarizer') {
                    formData.append('type', tool.subType);
                    response = await summarizeFile(formData);
                    setResult(response.data.summary);
                } else if (tool.type === 'solver' || tool.subType === 'image') {
                    // For image solvers
                    formData.append('subject', tool.subType);
                    response = await solveImageProblem(formData);
                    setResult(response?.result || response?.data || "No result returned");
                }
            }
            // 2. Text Generation / Solving
            else if (input) {
                if (tool.type === 'generator') {
                    response = await generateContent({
                        tool: "content_generator",
                        sub_tool: "generate",
                        topic: input,
                        content: input
                    });

                    setResult(response.result);
                } else if (tool.type === 'solver') {
                    response = await solveProblem({
                        tool: tool.type,
                        sub_tool: tool.subType,
                        content: input
                    });
                    setResult(response?.result || response?.data || "No result returned");
                }
            } else {
                setError('Please provide input or upload a file.');
            }

        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'An error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="glass-panel w-full max-w-2xl rounded-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
                {/* Header */}
                <div className="p-4 border-b border-slate-700/50 flex items-center justify-between bg-slate-900/50">
                    <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg bg-slate-800 ${tool.color}`}>
                            <tool.icon size={20} />
                        </div>
                        <h2 className="text-xl font-bold text-white">{tool.name}</h2>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-full transition-colors text-slate-400 hover:text-white">
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 overflow-y-auto custom-scrollbar flex-grow">
                    {!result ? (
                        <form onSubmit={handleSubmit} className="space-y-6">

                            {/* File Input */}
                            {(tool.input === 'file' || tool.input === 'mixed') && (
                                <div className="space-y-2">
                                    <label className="block text-sm font-medium text-slate-300">Upload File</label>
                                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl cursor-pointer bg-slate-800/30 hover:bg-slate-800/50 transition-all group">
                                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                            <Upload className="w-8 h-8 mb-3 text-slate-400 group-hover:text-sky-400 transition-colors" />
                                            <p className="text-sm text-slate-400">
                                                {file ? <span className="text-sky-400 font-medium">{file.name}</span> : <span>Click to upload or drag and drop</span>}
                                            </p>
                                            <p className="text-xs text-slate-500 mt-1">{tool.accept ? tool.accept.replace('.', '').toUpperCase() : 'Any'} files</p>
                                        </div>
                                        <input
                                            type="file"
                                            className="hidden"
                                            accept={tool.accept}
                                            onChange={(e) => setFile(e.target.files[0])}
                                        />
                                    </label>
                                </div>
                            )}

                            {/* Text Input */}
                            {(tool.input === 'text' || tool.input === 'mixed') && (
                                <div className="space-y-2">
                                    <label className="block text-sm font-medium text-slate-300">
                                        {tool.input === 'mixed' ? 'Or Enter Text' : 'Enter Prompt'}
                                    </label>
                                    <textarea
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        placeholder={tool.placeholder}
                                        className="w-full h-32 bg-slate-950/50 border border-slate-700/50 rounded-xl p-4 text-slate-200 placeholder-slate-600 focus:ring-2 focus:ring-sky-500/50 focus:border-transparent outline-none resize-none transition-all"
                                    />
                                </div>
                            )}

                            {error && (
                                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                                    {error}
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={loading || (!file && !input)}
                                className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold shadow-lg shadow-blue-500/20 flex items-center justify-center space-x-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="animate-spin" size={20} />
                                        <span>Processing...</span>
                                    </>
                                ) : (
                                    <>
                                        <Sparkles size={20} />
                                        <span>Generate</span>
                                    </>
                                )}
                            </button>
                        </form>
                    ) : (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-lg font-semibold text-sky-400">Result</h3>
                                <button
                                    onClick={() => setResult('')}
                                    className="text-sm text-slate-400 hover:text-white underline"
                                >
                                    Generate New
                                </button>
                            </div>
                            <div className="prose prose-invert prose-sm max-w-none prose-ai-response bg-slate-950/50 p-6 rounded-xl border border-slate-800">
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm, remarkMath]}
                                    rehypePlugins={[rehypeKatex, rehypeSanitize]}
                                    components={{
                                        p: ({ children }) => <p className="mb-4">{children}</p>,
                                        strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
                                        em: ({ children }) => <em className="italic text-slate-300">{children}</em>,
                                        ul: ({ children }) => <ul className="list-disc list-inside mb-4 space-y-1">{children}</ul>,
                                        ol: ({ children }) => <ol className="list-decimal list-inside mb-4 space-y-1">{children}</ol>,
                                        h1: ({ children }) => <h1 className="text-xl font-bold text-white mt-6 mb-3">{children}</h1>,
                                        h2: ({ children }) => <h2 className="text-lg font-bold text-white mt-5 mb-2">{children}</h2>,
                                        h3: ({ children }) => <h3 className="text-base font-bold text-white mt-4 mb-2">{children}</h3>,
                                        code: ({ className, children, ...props }) =>
                                            <code className="bg-slate-800 px-1.5 py-0.5 rounded text-sky-300 text-sm" {...props}>{children}</code>,
                                        pre: ({ children }) => <pre className="bg-slate-900 rounded-xl p-4 overflow-x-auto mb-4 border border-slate-700/50 whitespace-pre-wrap">{children}</pre>,
                                        blockquote: ({ children }) => <blockquote className="border-l-4 border-sky-500/50 pl-4 italic text-slate-300 mb-4">{children}</blockquote>,
                                    }}
                                >
                                    {typeof result === 'string' ? result : String(result)}
                                </ReactMarkdown>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ToolModal;
