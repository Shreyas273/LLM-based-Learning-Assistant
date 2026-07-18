import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSanitize from 'rehype-sanitize';
import { Bot, User, FileText, ChevronDown, ChevronUp } from 'lucide-react';

const ChatMessage = ({ message }) => {
    const isUser = message.role === 'user';
    const [showSources, setShowSources] = React.useState(false);

    const normalizedSources = React.useMemo(() => {
        const raw = Array.isArray(message.sources) ? message.sources : [];
        return raw
            .map((s) => {
                const pdf = s.pdf || s.pdf_name || s.source || 'Document';
                const page = (s.page ?? s.page_number ?? s.page) ?? null;
                const relevance = s.relevance_score ?? s.score ?? s.relevance ?? null;
                return { pdf, page, relevance };
            })
            .filter((s) => s.pdf || s.page != null || s.relevance != null);
    }, [message.sources]);

    return (
        <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex max-w-[85%] md:max-w-[75%] ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start gap-3`}>

                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${isUser ? 'bg-indigo-500/20 text-indigo-400' : 'bg-fuchsia-500/20 text-fuchsia-400'
                    }`}>
                    {isUser ? <User size={18} /> : <Bot size={18} />}
                </div>

                {/* Content */}
                <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
                    <div className={`px-4 py-3 rounded-2xl ${isUser
                        ? 'bg-slate-800 text-slate-100 rounded-tr-sm border border-slate-700/50'
                        : 'glass-panel text-slate-200 rounded-tl-sm'
                        }`}>
                        {message.image && (
                            <img src={message.image} alt="Upload" className="max-w-xs rounded-lg mb-2 border border-slate-700" />
                        )}

                        {/* Markdown Content - ChatGPT-style formatting */}
                        <div className="prose prose-invert prose-sm max-w-none prose-ai-response">
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
                                        className ? (
                                            <code className={`${className} bg-slate-800 px-1.5 py-0.5 rounded text-sky-300 text-sm`} {...props}>{children}</code>
                                        ) : (
                                            <code className="bg-slate-800 px-1.5 py-0.5 rounded text-sky-300 text-sm" {...props}>{children}</code>
                                        ),
                                    pre: ({ children }) => <pre className="bg-slate-900 rounded-xl p-4 overflow-x-auto mb-4 border border-slate-700/50">{children}</pre>,
                                    blockquote: ({ children }) => <blockquote className="border-l-4 border-sky-500/50 pl-4 italic text-slate-300 mb-4">{children}</blockquote>,
                                }}
                            >
                                {message.content}
                            </ReactMarkdown>
                        </div>

                        {/* Metadata Section */}
                        {!isUser && message.mode && (
                            <div className="mt-3 pt-3 border-t border-slate-700/50 flex flex-col gap-2">
                                <div className="flex flex-wrap items-center gap-2 text-xs">
                                    <span className={`px-2 py-0.5 rounded-full font-medium ${message.mode === 'RAG' ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'bg-slate-600/30 text-slate-400 border border-slate-600/30'
                                        }`}>
                                        {message.mode === 'RAG' ? '📚 PDF Context' : '🧠 General Knowledge'}
                                    </span>

                                    {message.confidence_text && (
                                        <span className={`px-2 py-0.5 rounded-full font-medium border ${(message.confidence || 0) > 60 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                                            (message.confidence || 0) > 30 ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                                                'bg-red-500/10 text-red-400 border-red-500/20'
                                            }`}>
                                            Confidence: {message.confidence_text}
                                        </span>
                                    )}
                                </div>

                                {/* Answer Source Viewer */}
                                {normalizedSources.length > 0 && (
                                    <div className="bg-slate-900/40 rounded-lg p-2 text-xs border border-slate-700/40">
                                        <button
                                            type="button"
                                            onClick={() => setShowSources((v) => !v)}
                                            className="w-full flex items-center justify-between text-slate-300 hover:text-slate-100 transition-colors"
                                        >
                                            <span className="flex items-center gap-1 font-medium text-slate-400">
                                                <FileText size={12} /> Answer sources ({normalizedSources.length})
                                            </span>
                                            {showSources ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                        </button>

                                        {showSources && (
                                            <div className="mt-2 space-y-1">
                                                {normalizedSources.map((src, idx) => (
                                                    <div
                                                        key={idx}
                                                        className="flex items-center justify-between gap-3 rounded-md px-2 py-1 bg-slate-950/30 border border-slate-800/50"
                                                    >
                                                        <div className="min-w-0">
                                                            <div className="text-slate-200 truncate">{src.pdf}</div>
                                                            <div className="text-[10px] text-slate-500">
                                                                {src.page != null ? `Page ${src.page}` : 'Page N/A'}
                                                            </div>
                                                        </div>
                                                        <div className="text-[10px] text-slate-400 flex-shrink-0">
                                                            {src.relevance != null ? `Relevance ${Math.round(Number(src.relevance))}%` : 'Relevance N/A'}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <span className="text-[10px] text-slate-500 mt-1 px-1">
                        {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default ChatMessage;
