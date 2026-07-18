import React, { useState, useRef } from 'react';
import { Send, Paperclip, X, Image as ImageIcon, Loader2 } from 'lucide-react';

const MODES = [
    { id: 'default', label: 'Normal' },
    { id: 'story', label: 'Story' },
    { id: 'flowchart', label: 'Flowchart' },
    { id: 'exam_prep', label: 'Exam Prep' },
];

const ChatInput = ({ onSend, loading, studyMode = 'default', onStudyModeChange }) => {
    const [input, setInput] = useState('');
    const [file, setFile] = useState(null);
    const fileInputRef = useRef(null);

    const handleSubmit = (e) => {
        e.preventDefault();
        if ((!input.trim() && !file) || loading) return;

        onSend(input, file);
        setInput('');
        setFile(null);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <div className="glass-panel p-4 rounded-2xl border-t border-slate-700/50">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="text-xs text-slate-400">Study mode</div>
                <div className="inline-flex rounded-xl border border-slate-700/60 bg-slate-950/40 p-1">
                    {MODES.map((m) => {
                        const active = m.id === studyMode;
                        return (
                            <button
                                key={m.id}
                                type="button"
                                onClick={() => onStudyModeChange?.(m.id)}
                                className={[
                                    "px-3 py-1.5 text-xs font-medium rounded-lg transition-all",
                                    active
                                        ? "bg-gradient-to-r from-sky-600/60 to-indigo-600/60 text-slate-100 shadow-sm shadow-sky-500/20"
                                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40",
                                ].join(" ")}
                            >
                                {m.label}
                            </button>
                        );
                    })}
                </div>
            </div>

            {file && (
                <div className="flex items-center gap-2 mb-3 bg-slate-800/50 p-2 rounded-lg w-fit border border-slate-700">
                    <div className="bg-sky-500/10 p-1.5 rounded text-sky-400">
                        <Paperclip size={14} />
                    </div>
                    <span className="text-sm text-slate-300 max-w-[200px] truncate">{file.name}</span>
                    <button
                        onClick={() => setFile(null)}
                        className="text-slate-500 hover:text-red-400 p-0.5 rounded-full transition-colors"
                    >
                        <X size={14} />
                    </button>
                </div>
            )}

            <form onSubmit={handleSubmit} className="flex items-end gap-3">
                <button
                    type="button"
                    onClick={() => {
                        fileInputRef.current?.click();
                    }}
                    className="p-3 rounded-xl transition-all text-slate-400 hover:text-sky-400 hover:bg-slate-800"
                >
                    <Paperclip size={20} />
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
                    />
                </button>

                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask anything about your documents..."
                    className="flex-grow bg-slate-950/50 border border-slate-700/50 rounded-xl p-3 max-h-32 min-h-[48px] resize-none focus:ring-1 focus:ring-sky-500 focus:border-transparent outline-none text-slate-200 placeholder-slate-600 custom-scrollbar"
                    rows={1}
                />

                <button
                    type="submit"
                    disabled={(!input.trim() && !file) || loading}
                    className="p-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                    {loading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
                </button>
            </form>
        </div>
    );
};

export default ChatInput;
