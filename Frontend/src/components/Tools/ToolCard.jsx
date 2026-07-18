import React from 'react';
import { ArrowRight } from 'lucide-react';

const ToolCard = ({ tool, onClick }) => {
    const Icon = tool.icon;

    return (
        <div
            onClick={() => onClick(tool)}
            className="glass-card p-6 rounded-2xl cursor-pointer group relative overflow-hidden"
        >
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

            <div className="relative z-10 flex flex-col items-start h-full">
                <div className={`p-3 rounded-xl bg-slate-800/50 mb-4 group-hover:scale-110 transition-transform duration-300 ${tool.color}`}>
                    <Icon size={24} />
                </div>

                <h3 className="text-lg font-semibold text-slate-200 mb-1 group-hover:text-white transition-colors">
                    {tool.name}
                </h3>

                <p className="text-sm text-slate-400 mb-4 flex-grow">
                    {tool.description}
                </p>

                <div className="flex items-center text-xs font-medium text-sky-400 opacity-0 group-hover:opacity-100 transition-all transform translate-y-2 group-hover:translate-y-0">
                    <span>Launch Tool</span>
                    <ArrowRight size={14} className="ml-1" />
                </div>
            </div>
        </div>
    );
};

export default ToolCard;
